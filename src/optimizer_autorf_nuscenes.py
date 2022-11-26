import random

import numpy as np
import torchvision
import torch
import torch.nn as nn
import json
from utils import get_rays_nuscenes, sample_from_rays, volume_rendering, volume_rendering2, image_float_to_uint8, rot_dist, generate_obj_sz_reg_samples, align_imgs_width
from skimage.metrics import structural_similarity as compute_ssim
from model_autorf import AutoRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
import os
import imageio
import time
import cv2
import pytorch3d.transforms.rotation_conversions as rot_trans


class OptimizerAutoRFNuScenes:
    def __init__(self, model_dir, gpu, nusc_dataset, jsonfile='srncar.json',
                 save_postfix='_nuscenes_autorf', num_workers=0, shuffle=False):
        """
        :param model_dir: the directory of pre-trained model
        :param gpu: which GPU we would use
        :param cam_id: the number of images for test-time optimization(ex : 000082.png)
        :param splits: test or val
        :param jsonfile: where the hyper-parameters are saved
        :param num_opts : number of test-time optimization steps
        """
        super().__init__()
        # Read Hyperparameters
        hpampath = os.path.join('jsonfiles', jsonfile)
        with open(hpampath, 'r') as f:
            self.hpams = json.load(f)
        self.save_postfix = save_postfix
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.load_model(model_dir)
        self.make_save_img_dir(model_dir)
        self.nusc_dataset = nusc_dataset
        self.dataloader = DataLoader(self.nusc_dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        print('we are going to save at ', self.save_dir)

        # initialize shapecode, texturecode, poses to optimize
        self.optimized_shapecodes = {}
        self.optimized_texturecodes = {}
        self.optimized_poses = {}
        self.optimized_ins_flag = {}
        self.optimized_ann_flag = {}
        for ii, anntoken in enumerate(self.nusc_dataset.anntokens):
            if anntoken not in self.optimized_poses.keys():
                self.optimized_poses[anntoken] = torch.zeros((3, 4), dtype=torch.float32)
                self.optimized_ann_flag[anntoken] = 0
        for ii, instoken in enumerate(self.nusc_dataset.instokens):
            if instoken not in self.optimized_shapecodes.keys():
                self.optimized_shapecodes[instoken] = None
                self.optimized_texturecodes[instoken] = None
                self.optimized_ins_flag[instoken] = 0

        # initialize evaluation scores
        self.psnr_eval = {}
        self.psnr_opt = {}
        self.ssim_eval = {}
        self.R_eval = {}
        self.T_eval = {}

    def optimize_objs(self, save_img=True, shapenet_obj_cood=True,
                      sym_aug=True):
        """
            Optimize on each annotation frame independently
        """

        # cam_ids = [ii for ii in range(0, self.num_cams_per_sample)]
        cam_ids = [0]
        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f'num obj: {batch_idx}/{len(self.dataloader)}')

            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            cam_poses = batch_data['cam_poses']
            valid_flags = batch_data['valid_flags']
            instoken = batch_data['instoken']
            anntoken = batch_data['anntoken']

            tgt_imgs, tgt_poses, masks_occ, rois, cam_intrinsics, valid_flags = \
                imgs[0, cam_ids], cam_poses[0, cam_ids], masks_occ[0, cam_ids], \
                rois[0, cam_ids], cam_intrinsics[0, cam_ids], valid_flags[0, cam_ids]

            if np.sum(valid_flags.numpy()) == 0:
                continue

            instoken, anntoken = instoken[0], anntoken[0]
            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            H, W = tgt_imgs.shape[1:3]
            rois[:, 0:2] -= self.hpams['roi_margin']
            rois[:, 2:4] += self.hpams['roi_margin']
            rois[:, 0:2] = torch.maximum(rois[:, 0:2], torch.as_tensor(0))
            rois[:, 2] = torch.minimum(rois[:, 2], torch.as_tensor(W - 1))
            rois[:, 3] = torch.minimum(rois[:, 3], torch.as_tensor(H - 1))

            tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[0], tgt_poses[0], masks_occ[0], rois[0], \
                                                  cam_intrinsics[0]

            # crop tgt img to roi
            tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
            mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ < 0)

            # preprocess image and predict shapecode and texturecode
            img_in = self.preprocess_img(tgt_img)
            shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
            shapecode = shapecode.detach().requires_grad_()
            texturecode = texturecode.detach().requires_grad_()

            # First Optimize
            self.nopts = 0
            self.set_optimizers(shapecode, texturecode)
            while self.nopts < self.hpams['optimize']['num_opts']:
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                loss_per_img = []

                # render ray values and prepare target rays
                rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = self.render_rays(tgt_img, mask_occ,
                                                                                             tgt_pose, obj_diag, K,
                                                                                             roi, shapecode,
                                                                                             texturecode,
                                                                                             shapenet_obj_cood,
                                                                                             sym_aug)

                # Compute losses
                # loss_rgb = torch.sum((rgb_rays - tgt_pixels) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                            torch.sum(torch.abs(occ_pixels)) + 1e-9)
                loss_occ = torch.sum(
                    torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                       torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                # loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ + self.hpams['loss_reg_coef'] * loss_reg
                # loss = loss_rgb + self.hpams['loss_reg_coef'] * loss_reg
                loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                loss.backward()
                loss_per_img.append(loss_rgb.detach().item())
                # Different roi sizes are dealt in save_image later
                gt_imgs.append(tgt_imgs[0, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                gt_masks_occ.append(masks_occ[0, roi[1]:roi[3], roi[0]:roi[2]])
                self.opts.step()

                # Just use the cropped region instead to save computation on the visualization
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        # for num, cam_id in enumerate(cam_ids):
                        tgt_pose, roi, K = tgt_poses[0], rois[0], cam_intrinsics[0]
                        # render full image
                        generated_img = self.render_full_img(tgt_pose, obj_sz, K, roi, shapecode, texturecode,
                                                             shapenet_obj_cood)
                        generated_imgs.append(generated_img)
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, anntoken, self.nopts)
                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            virtual_imgs = self.render_virtual_imgs(obj_sz, cam_intrinsics[0], shapecode, texturecode,
                                                                    shapenet_obj_cood)
                            self.save_virtual_img(virtual_imgs, anntoken, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers(shapecode, texturecode)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.optimized_ann_flag[anntoken] = 1
            self.save_opts(batch_idx)

    def optimize_objs_w_pose(self, save_img=True, shapenet_obj_cood=True,
                             sym_aug=True, obj_sz_reg=False, euler_rot=False):
        """
            Optimize on each annotation frame independently
        """

        # cam_ids = [ii for ii in range(0, self.num_cams_per_sample)]
        cam_ids = [0]
        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            # the dataloader is supposed to provide cam_poses with error (converted from object poses for nerf)
            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            cam_poses = batch_data['cam_poses']
            cam_poses_w_err = batch_data['cam_poses_w_err']
            valid_flags = batch_data['valid_flags']
            instoken = batch_data['instoken']
            anntoken = batch_data['anntoken']

            tgt_imgs, tgt_poses, pred_poses, masks_occ, rois, cam_intrinsics, valid_flags = \
                imgs[0, cam_ids], cam_poses[0, cam_ids], cam_poses_w_err[0, cam_ids], masks_occ[0, cam_ids], \
                rois[0, cam_ids], cam_intrinsics[0, cam_ids], valid_flags[0, cam_ids]

            if np.sum(valid_flags.numpy()) == 0:
                continue

            print(f'num obj: {batch_idx}/{len(self.dataloader)}')

            instoken, anntoken = instoken[0], anntoken[0]
            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            H, W = tgt_imgs.shape[1:3]
            rois[:, 0:2] -= self.hpams['roi_margin']
            rois[:, 2:4] += self.hpams['roi_margin']
            rois[:, 0:2] = torch.maximum(rois[:, 0:2], torch.as_tensor(0))
            rois[:, 2] = torch.minimum(rois[:, 2], torch.as_tensor(W - 1))
            rois[:, 3] = torch.minimum(rois[:, 3], torch.as_tensor(H - 1))

            tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[0], tgt_poses[0], masks_occ[0], rois[0], \
                                                  cam_intrinsics[0]

            # crop tgt img to roi
            tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
            mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ < 0)

            # preprocess image and predict shapecode and texturecode
            img_in = self.preprocess_img(tgt_img)
            shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
            shapecode = shapecode.detach().requires_grad_()
            texturecode = texturecode.detach().requires_grad_()

            # set pose parameters
            rot_mat_vec = pred_poses[:, :3, :3]
            trans_vec = pred_poses[:, :3, 3].to(self.device).detach().requires_grad_()
            if euler_rot:
                rot_vec = rot_trans.matrix_to_euler_angles(rot_mat_vec, 'XYZ').to(self.device).detach().requires_grad_()
            else:
                rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            # First Optimize
            self.nopts = 0
            self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            est_poses = torch.zeros((1, 3, 4), dtype=torch.float32)
            while self.nopts < self.hpams['optimize']['num_opts']:
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                loss_per_img = []

                t2opt = trans_vec[0].unsqueeze(-1)
                if euler_rot:
                    rot_mat2opt = rot_trans.euler_angles_to_matrix(rot_vec[0], 'XYZ')
                else:
                    rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[0])
                pose2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                # render ray values and prepare target rays
                rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = self.render_rays(tgt_img, mask_occ,
                                                                                             pose2opt, obj_diag, K,
                                                                                             roi, shapecode,
                                                                                             texturecode,
                                                                                             shapenet_obj_cood,
                                                                                             sym_aug)

                # Compute losses
                # Critical to let rgb supervised on white background
                # loss_rgb = torch.sum((rgb_rays - tgt_rays) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                            torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # Occupancy loss
                loss_occ = torch.sum(
                    torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                       torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                # loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ + self.hpams['loss_reg_coef'] * loss_reg
                # loss = loss_rgb + self.hpams['loss_reg_coef'] * loss_reg
                loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ

                if obj_sz_reg:
                    sz_reg_samples = generate_obj_sz_reg_samples(obj_sz, obj_diag, shapenet_obj_cood, tau=0.05)
                    loss_obj_sz = self.loss_obj_sz(sz_reg_samples, shapecode, texturecode)
                    loss = loss + self.hpams['loss_obj_sz_coef'] * loss_obj_sz

                loss.backward()
                loss_per_img.append(loss_rgb.detach().item())
                # Different roi sizes are dealt in save_image later
                gt_imgs.append(tgt_imgs[0, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                gt_masks_occ.append(masks_occ[0, roi[1]:roi[3], roi[0]:roi[2]])
                est_poses[0] = pose2opt.detach().cpu()
                errs_R, errs_T = self.calc_obj_pose_err(est_poses, tgt_poses)
                if self.nopts == 0:
                    print('   Initial R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                    print('   Final R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                self.opts.step()

                # Just use the cropped region instead to save computation on the visualization
                # ATTENTION: the first visual is already after one iter of optimization
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        # for num, cam_id in enumerate(cam_ids):
                        tgt_pose, roi, K = tgt_poses[0], rois[0], cam_intrinsics[0]
                        pose2opt = est_poses[0]
                        # render full image
                        generated_img = self.render_full_img(pose2opt, obj_sz, K, roi, shapecode, texturecode,
                                                             shapenet_obj_cood, debug_occ=False)
                        # mark pose error on the image
                        err_str = 'R err: {:.3f}, T err: {:.3f}'.format(errs_R[0], errs_T[0])
                        generated_img = cv2.putText(generated_img.cpu().numpy(), err_str, (5, 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, .3, (1, 0, 0))
                        generated_imgs.append(torch.from_numpy(generated_img))
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, anntoken, self.nopts)

                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            virtual_imgs = self.render_virtual_imgs(obj_sz, cam_intrinsics[0], shapecode, texturecode,
                                                                    shapenet_obj_cood)
                            self.save_virtual_img(virtual_imgs, anntoken, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
                    # self.set_optimizers_w_poses_model(shapecode, texturecode, rot_vec, trans_vec)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.optimized_ann_flag[anntoken] = 1
            self.log_eval_pose(est_poses, tgt_poses, batch_data['anntoken'])
            self.save_opts_w_pose(batch_idx)

    def optimize_objs_multi_anns(self, save_img=True, shapenet_obj_cood=True, sym_aug=True):
        """
            optimize multiple annotations for the same instance in a singe iteration
        """

        instokens = self.nusc_dataset.ins_ann_tokens.keys()

        # Per object
        for obj_idx, instoken in enumerate(instokens):
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}')

            tgt_imgs, masks_occ, rois, cam_intrinsics, tgt_poses, anntokens = self.nusc_dataset.get_ins_samples(
                instoken)

            if tgt_imgs is None:
                continue

            print(f'    num views: {tgt_imgs.shape[0]}')
            H, W = tgt_imgs.shape[1:3]
            rois[..., 0:2] -= self.hpams['roi_margin']
            rois[..., 2:4] += self.hpams['roi_margin']
            rois[..., 0:2] = torch.maximum(rois[..., 0:2], torch.as_tensor(0))
            rois[..., 2] = torch.minimum(rois[..., 2], torch.as_tensor(W - 1))
            rois[..., 3] = torch.minimum(rois[..., 3], torch.as_tensor(H - 1))

            # compute the mean shapecode and texturecode from different views
            shapecode_list, texturecode_list = [], []
            for num in range(0, tgt_imgs.shape[0]):
                tgt_img, mask_occ, roi = tgt_imgs[num], masks_occ[num], rois[num]

                # crop tgt img to roi
                tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                tgt_img = tgt_img * (mask_occ > 0)
                tgt_img = tgt_img + (mask_occ < 0)

                # preprocess image and predict shapecode and texturecode
                img_in = self.preprocess_img(tgt_img)
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                shapecode_list.append(shapecode)
                texturecode_list.append(texturecode)
            shapecode = torch.mean(torch.cat(shapecode_list), dim=0, keepdim=True).detach().requires_grad_()
            texturecode = torch.mean(torch.cat(texturecode_list), dim=0, keepdim=True).detach().requires_grad_()

            # Optimize
            self.nopts = 0
            self.set_optimizers(shapecode, texturecode)
            while self.nopts < self.hpams['optimize']['num_opts']:
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                loss_per_img = []
                for num in range(0, tgt_imgs.shape[0]):
                    tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], tgt_poses[num], masks_occ[num], rois[num], \
                                                          cam_intrinsics[num]

                    obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

                    # crop tgt img to roi
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                    # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                    tgt_img = tgt_img * (mask_occ > 0)
                    tgt_img = tgt_img + (mask_occ < 0)

                    # render ray values and prepare target rays
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = self.render_rays(tgt_img, mask_occ,
                                                                                                 tgt_pose, obj_diag, K,
                                                                                                 roi, shapecode,
                                                                                                 texturecode,
                                                                                                 shapenet_obj_cood,
                                                                                                 sym_aug)

                    # Compute losses
                    # loss_rgb = torch.sum((rgb_rays - tgt_rays) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                    loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                                torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # Occupancy loss
                    loss_occ = torch.sum(
                        torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                           torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                    # loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ + self.hpams['loss_reg_coef'] * loss_reg
                    # loss = loss_rgb + self.hpams['loss_reg_coef'] * loss_reg
                    loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                    loss.backward()
                    loss_per_img.append(loss_rgb.detach().item())

                    # Different roi sizes are dealt in save_image later
                    gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])
                    self.optimized_ann_flag[anntokens[num]] = 1

                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        for num in range(0, tgt_imgs.shape[0]):
                            tgt_pose, roi, K = tgt_poses[num], rois[num], cam_intrinsics[num]
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                            # render full image
                            generated_img = self.render_full_img(tgt_pose, obj_sz, K, roi, shapecode, texturecode,
                                                                 shapenet_obj_cood)
                            generated_imgs.append(generated_img)
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, instoken, self.nopts)

                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[0])['size']
                            virtual_imgs = self.render_virtual_imgs(obj_sz, cam_intrinsics[0], shapecode, texturecode,
                                                                    shapenet_obj_cood)
                            self.save_virtual_img(virtual_imgs, instoken, self.nopts)

                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers(shapecode, texturecode)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.save_opts(obj_idx)

    def optimize_objs_multi_anns_w_pose(self, save_img=True,
                                        shapenet_obj_cood=True, sym_aug=True, euler_rot=False):
        """
            optimize multiple annotations for the same instance in a singe iteration
        """

        instokens = self.nusc_dataset.ins_ann_tokens.keys()
        # Per object
        for obj_idx, instoken in enumerate(instokens):
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}')

            tgt_imgs, masks_occ, rois, cam_intrinsics, tgt_poses, pred_poses, anntokens = self.nusc_dataset.get_ins_samples(
                instoken)

            if tgt_imgs is None:
                continue

            # if tgt_imgs is None or tgt_imgs.shape[0] < 2:
            #     continue

            print(f'    num views: {tgt_imgs.shape[0]}')
            H, W = tgt_imgs.shape[1:3]
            rois[..., 0:2] -= self.hpams['roi_margin']
            rois[..., 2:4] += self.hpams['roi_margin']
            rois[..., 0:2] = torch.maximum(rois[..., 0:2], torch.as_tensor(0))
            rois[..., 2] = torch.minimum(rois[..., 2], torch.as_tensor(W - 1))
            rois[..., 3] = torch.minimum(rois[..., 3], torch.as_tensor(H - 1))

            # compute the mean shapecode and texturecode from different views
            shapecode_list, texturecode_list = [], []
            for num in range(0, tgt_imgs.shape[0]):
                tgt_img, mask_occ, roi = tgt_imgs[num], masks_occ[num], rois[num]

                # crop tgt img to roi
                tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                tgt_img = tgt_img * (mask_occ > 0)
                tgt_img = tgt_img + (mask_occ < 0)

                # preprocess image and predict shapecode and texturecode
                img_in = self.preprocess_img(tgt_img)
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                shapecode_list.append(shapecode)
                texturecode_list.append(texturecode)
            shapecode = torch.mean(torch.cat(shapecode_list), dim=0, keepdim=True).detach().requires_grad_()
            texturecode = torch.mean(torch.cat(texturecode_list), dim=0, keepdim=True).detach().requires_grad_()

            # set pose parameters
            rot_mat_vec = pred_poses[:, :3, :3]
            trans_vec = pred_poses[:, :3, 3].to(self.device).detach().requires_grad_()
            if euler_rot:
                rot_vec = rot_trans.matrix_to_euler_angles(rot_mat_vec, 'XYZ').to(self.device).detach().requires_grad_()
            else:
                rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            # Optimize
            self.nopts = 0
            self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            est_poses = torch.zeros((tgt_imgs.shape[0], 3, 4), dtype=torch.float32)
            while self.nopts < self.hpams['optimize']['num_opts']:
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                loss_per_img = []
                for num in range(0, tgt_imgs.shape[0]):
                    tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], tgt_poses[num], masks_occ[num], rois[num], \
                                                          cam_intrinsics[num]

                    t2opt = trans_vec[num].unsqueeze(-1)
                    if euler_rot:
                        rot_mat2opt = rot_trans.euler_angles_to_matrix(rot_vec[num], 'XYZ')
                    else:
                        rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[num])
                    pose2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                    obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

                    # crop tgt img to roi
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                    # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                    tgt_img = tgt_img * (mask_occ > 0)
                    tgt_img = tgt_img + (mask_occ < 0)

                    # render ray values and prepare target rays
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = self.render_rays(tgt_img, mask_occ,
                                                                                                 pose2opt, obj_diag, K,
                                                                                                 roi,
                                                                                                 shapecode,
                                                                                                 texturecode,
                                                                                                 shapenet_obj_cood,
                                                                                                 sym_aug)

                    # Compute losses
                    # loss_rgb = torch.sum((rgb_rays - tgt_rays) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                    loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                                torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # Occupancy loss
                    loss_occ = torch.sum(
                        torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                           torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                    # loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ + self.hpams['loss_reg_coef'] * loss_reg
                    # loss = loss_rgb + self.hpams['loss_reg_coef'] * loss_reg
                    loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                    loss.backward()
                    loss_per_img.append(loss_rgb.detach().item())

                    # Different roi sizes are dealt in save_image later
                    gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])
                    self.optimized_ann_flag[anntokens[num]] = 1
                    est_poses[num] = pose2opt.detach().cpu()
                errs_R, errs_T = self.calc_obj_pose_err(est_poses, tgt_poses)
                if self.nopts == 0:
                    print('    Initial R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                    print('    Final R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []

                    with torch.no_grad():
                        for num in range(0, tgt_imgs.shape[0]):
                            tgt_pose, roi, K = tgt_poses[num], rois[num], cam_intrinsics[num]
                            pose2opt = est_poses[num]
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                            # render full image
                            generated_img = self.render_full_img(pose2opt, obj_sz, K, roi, shapecode, texturecode,
                                                                 shapenet_obj_cood)
                            # mark pose error on the image
                            err_str = 'R err: {:.3f}, T err: {:.3f}'.format(errs_R[num], errs_T[num])
                            generated_img = cv2.putText(generated_img.cpu().numpy(), err_str, (5, 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, .3, (1, 0, 0))
                            generated_imgs.append(torch.from_numpy(generated_img))
                            # save the last pose for later evaluation
                            if self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                                est_poses[num] = pose2opt.detach().cpu()
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, instoken, self.nopts)

                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[0])['size']
                            virtual_imgs = self.render_virtual_imgs(obj_sz, cam_intrinsics[0], shapecode, texturecode,
                                                                    shapenet_obj_cood)
                            self.save_virtual_img(virtual_imgs, instoken, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.log_eval_pose(est_poses, tgt_poses, anntokens)
            self.save_opts_w_pose(obj_idx)

    def loss_obj_sz(self, sz_reg_samples, shapecode, texturecode):
        samples_out = np.concatenate((np.expand_dims(sz_reg_samples['X_planes_out'], 0),
                                    np.expand_dims(sz_reg_samples['Y_planes_out'], 0),
                                    np.expand_dims(sz_reg_samples['Z_planes_out'], 0)), axis=0)
        samples_out = torch.from_numpy(samples_out)
        samples_in = np.concatenate((np.expand_dims(sz_reg_samples['X_planes_in'], 0),
                                    np.expand_dims(sz_reg_samples['Y_planes_in'], 0),
                                    np.expand_dims(sz_reg_samples['Z_planes_in'], 0)), axis=0)
        samples_in = torch.from_numpy(samples_in)

        sigmas_out, _ = self.model(samples_out.to(self.device),
                                 torch.ones_like(samples_out).to(self.device),
                                 shapecode, texturecode)
        sigmas_in, _ = self.model(samples_in.to(self.device),
                                 torch.ones_like(samples_in).to(self.device),
                                 shapecode, texturecode)
        sigmas_out_max = torch.max(sigmas_out.squeeze(), dim=1).values
        sigmas_in_max = torch.max(sigmas_in.squeeze(), dim=1).values

        loss = torch.sum(sigmas_out_max ** 2) + \
               torch.sum((sigmas_in_max - torch.ones_like(sigmas_in_max))**2)

        return loss / 6

    def save_opts(self, num_obj):
        saved_dict = {
            'num_obj': num_obj,
            'optimized_shapecodes': self.optimized_shapecodes,
            'optimized_texturecodes': self.optimized_texturecodes,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval,
            'optimized_ins_flag': self.optimized_ins_flag,
            'optimized_ann_flag': self.optimized_ann_flag,
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes.pth'))
        # print('We finished the optimization of object' + str(num_obj))

    def save_opts_w_pose(self, num_obj):
        saved_dict = {
            'num_obj': num_obj,
            'optimized_shapecodes': self.optimized_shapecodes,
            'optimized_texturecodes': self.optimized_texturecodes,
            'optimized_poses': self.optimized_poses,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval,
            'R_eval': self.R_eval,
            'T_eval': self.T_eval,
            'optimized_ins_flag': self.optimized_ins_flag,
            'optimized_ann_flag': self.optimized_ann_flag,
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes+poses.pth'))
        # print('We finished the optimization of ' + str(num_obj))

    def preprocess_img(self, img):
        img = img.unsqueeze(0).permute((0, 3, 1, 2))
        _, _, im_h, im_w = img.shape
        if np.maximum(im_h, im_w) > self.hpams['max_img_sz']:
            ratio = self.hpams['max_img_sz'] / np.maximum(im_h, im_w)
            new_h = im_h * ratio
            new_w = im_w * ratio
            img = Resize((int(new_h), int(new_w)))(img)
        return img

    def render_rays(self, img, mask_occ, cam_pose, obj_diag, K, roi, shapecode, texturecode, shapenet_obj_cood, sym_aug):
        # near and far sample range need to be adaptively calculated
        near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
        far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2

        rays_o, viewdir = get_rays_nuscenes(K, cam_pose, roi)

        # For different sized roi, extract a random subset of pixels with fixed batch size
        n_rays = np.minimum(rays_o.shape[0], self.hpams['n_rays'])
        random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
        rays_o = rays_o[random_ray_ids]
        viewdir = viewdir[random_ray_ids]

        # # The random selection should be within the roi pixels
        # img = img[roi[1]: roi[3], roi[0]: roi[2]]
        # mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
        #
        # # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
        # img = img * (mask_occ > 0)
        # img = img + (mask_occ < 0)

        # extract samples
        rgb_tgt = img.reshape(-1, 3)[random_ray_ids].to(self.device)
        occ_pixels = mask_occ.reshape(-1, 1)[random_ray_ids].to(self.device)
        mask_rgb = torch.clone(mask_occ)
        mask_rgb[mask_rgb < 0] = 0

        xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, self.hpams['n_samples'])
        xyz /= obj_diag

        # Apply symmetric augmentation
        if sym_aug and random.uniform(0, 1) > 0.5:
            xyz[:, :, 1] *= (-1)
            viewdir[:, :, 1] *= (-1)

        # Nuscene to ShapeNet: frame rotate -90 degree around Z, coord rotate 90 degree around Z
        if shapenet_obj_cood:
            xyz = xyz[:, :, [1, 0, 2]]
            xyz[:, :, 0] *= (-1)
            viewdir = viewdir[:, :, [1, 0, 2]]
            viewdir[:, :, 0] *= (-1)

        sigmas, rgbs = self.model(xyz.to(self.device),
                                  viewdir.to(self.device),
                                  shapecode, texturecode)
        rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(self.device))
        return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels

    def render_full_img(self, cam_pose, obj_sz, K, roi, shapecode, texturecode, shapenet_obj_cood, debug_occ=False):
        obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

        # near and far sample range need to be adaptively calculated
        near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
        far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2

        rays_o, viewdir = get_rays_nuscenes(K, cam_pose, roi)
        xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, self.hpams['n_samples'])
        xyz /= obj_diag
        # Nuscene to ShapeNet: rotate 90 degree around Z
        if shapenet_obj_cood:
            xyz = xyz[:, :, [1, 0, 2]]
            xyz[:, :, 0] *= (-1)
            viewdir = viewdir[:, :, [1, 0, 2]]
            viewdir[:, :, 0] *= (-1)

        generated_img = []
        generated_acc_trans = []
        sample_step = np.maximum(roi[2] - roi[0], roi[3] - roi[1])
        for i in range(0, xyz.shape[0], sample_step):
            sigmas, rgbs = self.model(xyz[i:i + sample_step].to(self.device),
                                      viewdir[i:i + sample_step].to(self.device),
                                      shapecode, texturecode)
            if debug_occ:
                rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(self.device))
                generated_acc_trans.append(acc_trans_rays)
            else:
                rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
            generated_img.append(rgb_rays)
        generated_img = torch.cat(generated_img).reshape(roi[3] - roi[1], roi[2] - roi[0], 3)

        if debug_occ:
            generated_acc_trans = torch.cat(generated_acc_trans).reshape(roi[3]-roi[1], roi[2]-roi[0])
            cv2.imshow('est_occ', ((torch.ones_like(generated_acc_trans) - generated_acc_trans).cpu().numpy() * 255).astype(np.uint8))
            # cv2.imshow('mask_occ', ((gt_masks_occ[0].cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8))
            cv2.waitKey()

        return generated_img

    def render_virtual_imgs(self, obj_sz, K, shapecode, texturecode, shapenet_obj_cood, radius=40., tilt=np.pi/6, pan_num=8, img_sz=128):
        virtual_imgs = []
        x_min = K[0, 2] - img_sz/2
        x_max = K[0, 2] + img_sz/2
        y_min = K[1, 2] - img_sz/2
        y_max = K[1, 2] + img_sz/2
        roi = np.asarray([x_min, y_min, x_max, y_max]).astype(np.int)
        # sample camera with fixed radius, tilt, and pan angles spanning 2 pi
        cam_init = np.asarray([[0,   0,  1, -radius],
                               [-1,  0,  0, 0],
                               [0,  -1,  0, 0],
                               [0,   0,  0, 1]]).astype(np.float32)
        cam_tilt = np.asarray([[np.cos(tilt),   0, np.sin(tilt), 0],
                               [0,              1, 0,             0],
                               [-np.sin(tilt),   0, np.cos(tilt),  0],
                               [0,              0, 0,             1]]).astype(np.float32) @ cam_init

        pan_angles = np.linspace(0, 2*np.pi, pan_num, endpoint=False)
        for pan in pan_angles:
            cam_pose = np.asarray([[np.cos(pan),   -np.sin(pan), 0, 0],
                                   [np.sin(pan),   np.cos(pan),  0, 0],
                                   [0,              0,           1, 0],
                                   [0,              0,           0, 1]]).astype(np.float32) @ cam_tilt
            cam_pose = torch.from_numpy(cam_pose[:3, :])
            generated_img = self.render_full_img(cam_pose, obj_sz, K, roi, shapecode, texturecode, shapenet_obj_cood)
            # draw the object coordinate basis
            R_w2c = cam_pose[:3, :3].transpose(-1, -2)
            T_w2c = -torch.matmul(R_w2c, cam_pose[:3, 3:])
            P_w2c = torch.cat((R_w2c, T_w2c), dim=1).numpy()
            x_arrow_2d = K @ P_w2c @ torch.asarray([.5, 0., 0., 1.]).reshape([-1, 1])
            x_arrow_2d = (x_arrow_2d[:2] / x_arrow_2d[2]).squeeze().numpy() - K[:2, 2].numpy()
            y_arrow_2d = K @ P_w2c @ torch.asarray([0., .5, 0., 1.]).reshape([-1, 1])
            y_arrow_2d = (y_arrow_2d[:2] / y_arrow_2d[2]).squeeze().numpy() - K[:2, 2].numpy()
            z_arrow_2d = K @ P_w2c @ torch.asarray([0., 0., .5, 1.]).reshape([-1, 1])
            z_arrow_2d = (z_arrow_2d[:2] / z_arrow_2d[2]).squeeze().numpy() - K[:2, 2].numpy()
            generated_img = generated_img.cpu().numpy()
            generated_img = cv2.arrowedLine(generated_img,
                                            (int(img_sz / 2), int(img_sz / 2)),
                                            (int(img_sz/2 + x_arrow_2d[0]), int(img_sz/2 + x_arrow_2d[1])),
                                            (1, 0, 0))
            generated_img = cv2.arrowedLine(generated_img,
                                            (int(img_sz / 2), int(img_sz / 2)),
                                            (int(img_sz/2 + y_arrow_2d[0]), int(img_sz/2 + y_arrow_2d[1])),
                                            (0, 1, 0))
            generated_img = cv2.arrowedLine(generated_img,
                                            (int(img_sz / 2), int(img_sz / 2)),
                                            (int(img_sz/2 + z_arrow_2d[0]), int(img_sz/2 + z_arrow_2d[1])),
                                            (0, 0, 1))
            virtual_imgs.append(torch.from_numpy(generated_img))

        return virtual_imgs

    def save_img(self, generated_imgs, gt_imgs, masks_occ, obj_id, instance_num):
        # H, W = gt_imgs[0].shape[:2]
        W_tgt = np.min([gt_img.shape[1] for gt_img in gt_imgs])
        # nviews = len(gt_imgs)

        if len(gt_imgs) > 1:
            # Align the width of different-sized images
            generated_imgs = align_imgs_width(generated_imgs, W_tgt)
            masks_occ = align_imgs_width(masks_occ, W_tgt)
            gt_imgs = align_imgs_width(gt_imgs, W_tgt)

        generated_imgs = torch.cat(generated_imgs).reshape(-1, W_tgt, 3)
        masks_occ = torch.cat(masks_occ).reshape(-1, W_tgt, 1)
        gt_imgs = torch.cat(gt_imgs).reshape(-1, W_tgt, 3)
        H_cat = generated_imgs.shape[0]

        ret = torch.zeros(H_cat, 2 * W_tgt, 3)
        ret[:, :W_tgt, :] = generated_imgs.reshape(-1, W_tgt, 3)
        ret[:, W_tgt:, :] = gt_imgs.reshape(-1, W_tgt, 3) * 0.75 + masks_occ.reshape(-1, W_tgt, 1) * 0.25
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        # imageio.imwrite(os.path.join(save_img_dir, 'opt' + self.nviews + '_{:03d}'.format(instance_num) + '.png'), ret)
        imageio.imwrite(os.path.join(save_img_dir, 'opt' + '{:03d}'.format(instance_num) + '.png'), ret)

    def save_virtual_img(self, imgs, obj_id, instance_num=None):
        H, W = imgs[0].shape[:2]
        # nviews = len(gt_imgs)

        img_out = torch.cat(imgs).reshape(-1, W, 3)
        img_out = image_float_to_uint8(img_out.detach().cpu().numpy())
        img_out = np.concatenate([img_out[:4*H, ...], img_out[4*H:, ...]], axis=1)
        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        if instance_num is None:
            imageio.imwrite(os.path.join(save_img_dir, 'virt_final.png'), img_out)
        else:
            imageio.imwrite(os.path.join(save_img_dir, 'virt_opt' + '{:03d}'.format(instance_num) + '.png'), img_out)

    def log_compute_ssim(self, generated_img, gt_img, niters, obj_idx):
        generated_img_np = generated_img.detach().cpu().numpy()
        gt_img_np = gt_img.detach().cpu().numpy()
        ssim = compute_ssim(generated_img_np, gt_img_np, multichannel=True)
        # if niters == 0:
        if self.ssim_eval.get(obj_idx) is None:
            self.ssim_eval[obj_idx] = [ssim]
        else:
            self.ssim_eval[obj_idx].append(ssim)

    def log_eval_psnr(self, loss_per_img, obj_idx):
        psnr = -10 * np.log(loss_per_img) / np.log(10)
        # if niters == 0:
        if self.psnr_eval.get(obj_idx) is None:
            self.psnr_eval[obj_idx] = [psnr]
        else:
            self.psnr_eval[obj_idx].append(psnr)

    def log_eval_pose(self, est_poses, tgt_poses, ann_tokens):
        # convert back to object pose to evaluaate
        est_R = est_poses[:, :3, :3].transpose(-1, -2)
        est_T = -torch.matmul(est_R, est_poses[:, :3, 3:]).squeeze(-1)
        tgt_R = tgt_poses[:, :3, :3].transpose(-1, -2)
        tgt_T = -torch.matmul(tgt_R, tgt_poses[:, :3, 3:]).squeeze(-1)

        err_R = rot_dist(est_R, tgt_R)
        err_T = torch.sqrt(torch.sum((est_T - tgt_T)**2, dim=-1))

        for i, ann_token in enumerate(ann_tokens):
            self.R_eval[ann_token] = err_R[i]
            self.T_eval[ann_token] = err_T[i]
            # print(f'    R_error: {self.R_eval[ann_token]}, T_error: {self.T_eval[ann_token]}')

    def calc_cam_pose_err(self, est_poses, tgt_poses):
        est_R = est_poses[:, :3, :3]
        est_T = est_poses[:, :3, 3]
        tgt_R = tgt_poses[:, :3, :3]
        tgt_T = tgt_poses[:, :3, 3]

        err_R = rot_dist(est_R, tgt_R)
        err_T = torch.sqrt(torch.sum((est_T - tgt_T) ** 2, dim=-1))
        return err_R, err_T

    def calc_obj_pose_err(self, est_poses, tgt_poses):
        est_R = est_poses[:, :3, :3].transpose(-1, -2)
        est_T = -torch.matmul(est_R, est_poses[:, :3, 3:]).squeeze(-1)
        tgt_R = tgt_poses[:, :3, :3].transpose(-1, -2)
        tgt_T = -torch.matmul(tgt_R, tgt_poses[:, :3, 3:]).squeeze(-1)

        err_R = rot_dist(est_R, tgt_R)
        err_T = torch.sqrt(torch.sum((est_T - tgt_T) ** 2, dim=-1))
        return err_R, err_T

    def set_optimizers(self, shapecode, texturecode):
        self.update_learning_rate()
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': self.hpams['optimize']['lr_shape']},
            {'params': texturecode, 'lr': self.hpams['optimize']['lr_texture']}
        ])

    def set_optimizers_w_poses(self, shapecode, texturecode, rots, trans):
        self.update_learning_rate()
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': self.hpams['optimize']['lr_shape']},
            {'params': texturecode, 'lr': self.hpams['optimize']['lr_texture']},
            {'params': rots, 'lr': self.hpams['optimize']['lr_pose']},
            {'params': trans, 'lr':  self.hpams['optimize']['lr_pose']}
        ])

    def update_learning_rate(self):
        opt_values = self.nopts // self.hpams['optimize']['lr_half_interval']
        self.hpams['optimize']['lr_shape'] = self.hpams['optimize']['lr_shape'] * 2**(-opt_values)
        self.hpams['optimize']['lr_texture'] = self.hpams['optimize']['lr_texture'] * 2**(-opt_values)
        self.hpams['optimize']['lr_pose'] = self.hpams['optimize']['lr_pose'] * 2**(-opt_values)

    def make_model(self):
        self.model = AutoRF(**self.hpams['net_hyperparams']).to(self.device)

    def load_model(self, saved_dir):
        saved_path = os.path.join(saved_dir, 'models.pth')
        saved_data = torch.load(saved_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)

    def make_save_img_dir(self, save_dir):
        save_dir_tmp = save_dir + '/test' + self.save_postfix
        num = 2
        while os.path.isdir(save_dir_tmp):
            save_dir_tmp = save_dir + '/test' + self.save_postfix + '_' + str(num)
            num += 1

        os.makedirs(save_dir_tmp)
        self.save_dir = save_dir_tmp
