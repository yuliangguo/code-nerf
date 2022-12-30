import os
import imageio
import time
import cv2
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import pytorch3d.transforms.rotation_conversions as rot_trans

from utils import image_float_to_uint8, rot_dist, generate_obj_sz_reg_samples, align_imgs_width, render_rays, render_full_img, render_virtual_imgs, calc_obj_pose_err,  preprocess_img_square
from skimage.metrics import structural_similarity as compute_ssim
from model_autorf import AutoRF
from model_codenerf import CodeNeRF


class OptimizerNuScenes:
    def __init__(self, gpu, nusc_dataset, hpams,
                 save_postfix='_nuscenes', num_workers=0, shuffle=False):
        super().__init__()
        self.hpams = hpams
        self.save_postfix = save_postfix
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.load_model()
        self.make_save_img_dir()
        self.nusc_dataset = nusc_dataset
        self.dataloader = DataLoader(self.nusc_dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        print('we are going to save at ', self.save_dir)

        # initialize shapecode, texturecode, poses to optimize
        self.optimized_shapecodes = {}
        self.optimized_texturecodes = {}
        self.optimized_poses = {}
        self.optimized_ins_flag = {}
        self.optimized_ann_flag = {}
        for ii, anntoken in enumerate(self.nusc_dataset.instoken_per_ann.keys()):
            if anntoken not in self.optimized_poses.keys():
                self.optimized_poses[anntoken] = torch.zeros((3, 4), dtype=torch.float32)
                self.optimized_ann_flag[anntoken] = 0
        for ii, instoken in enumerate(self.nusc_dataset.anntokens_per_ins.keys()):
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

    def optimize_objs(self, save_img=True):
        """
            Optimize on each annotation frame independently
        """

        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f'num obj: {batch_idx}/{len(self.dataloader)}')

            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            cam_poses = batch_data['cam_poses']
            instoken = batch_data['instoken']
            anntoken = batch_data['anntoken']

            tgt_img, tgt_cam, mask_occ, roi, K = \
                imgs[0], cam_poses[0], masks_occ[0], rois[0], cam_intrinsics[0]

            instoken, anntoken = instoken[0], anntoken[0]
            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            H, W = tgt_img.shape[0:2]
            roi[0:2] -= self.hpams['roi_margin']
            roi[2:4] += self.hpams['roi_margin']
            roi[0:2] = torch.maximum(roi[0:2], torch.as_tensor(0))
            roi[2] = torch.minimum(roi[2], torch.as_tensor(W - 1))
            roi[3] = torch.minimum(roi[3], torch.as_tensor(H - 1))

            # crop tgt img to roi
            tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
            mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ < 0)

            if self.hpams['arch'] == 'autorf':
                # preprocess image and predict shapecode and texturecode
                # img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                shapecode = shapecode.detach().requires_grad_()
                texturecode = texturecode.detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')

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
                rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays(self.model, self.device,
                                                                                        tgt_img, mask_occ, tgt_cam,
                                                                                        obj_diag, K, roi,
                                                                                        self.hpams['n_rays'],
                                                                                        self.hpams['n_samples'],
                                                                                        shapecode, texturecode,
                                                                                        self.hpams['shapenet_obj_cood'],
                                                                                        self.hpams['sym_aug'])

                # Compute losses
                # loss_rgb = torch.sum((rgb_rays - tgt_pixels) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                            torch.sum(torch.abs(occ_pixels)) + 1e-9)
                loss_occ = torch.sum(
                    torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                       torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                loss.backward()
                loss_per_img.append(loss_rgb.detach().item())
                # Different roi sizes are dealt in save_image later
                gt_imgs.append(imgs[0, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                gt_masks_occ.append(masks_occ[0, roi[1]:roi[3], roi[0]:roi[2]])
                if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                    self.log_eval_psnr(loss_per_img, batch_data['anntoken'])
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        # for num, cam_id in enumerate(cam_ids):
                        # render full image
                        generated_img = render_full_img(self.model, self.device, tgt_cam, obj_sz, K, roi,
                                                        self.hpams['n_samples'], shapecode, texturecode,
                                                        self.hpams['shapenet_obj_cood'])
                        generated_imgs.append(generated_img)
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, anntoken, self.nopts)
                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                                                               self.hpams['n_samples'], shapecode, texturecode,
                                                               self.hpams['shapenet_obj_cood'])
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

    def optimize_objs_w_pose(self, save_img=True):
        """
            Optimize on each annotation frame independently
        """

        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f'num obj: {batch_idx}/{len(self.dataloader)}')
            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            cam_poses = batch_data['cam_poses']
            cam_poses_w_err = batch_data['cam_poses_w_err']
            obj_poses_w_err = batch_data['obj_poses_w_err']
            instoken = batch_data['instoken']
            anntoken = batch_data['anntoken']

            tgt_img, tgt_cam, pred_pose, mask_occ, roi, K = \
                imgs[0], cam_poses[0], cam_poses_w_err[0], masks_occ[0], \
                rois[0], cam_intrinsics[0]

            if self.hpams['optimize']['opt_obj_pose']:
                pred_pose = obj_poses_w_err[0]

            instoken, anntoken = instoken[0], anntoken[0]
            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            H, W = tgt_img.shape[0:2]
            roi[0:2] -= self.hpams['roi_margin']
            roi[2:4] += self.hpams['roi_margin']
            roi[0:2] = torch.maximum(roi[0:2], torch.as_tensor(0))
            roi[2] = torch.minimum(roi[2], torch.as_tensor(W - 1))
            roi[3] = torch.minimum(roi[3], torch.as_tensor(H - 1))

            # crop tgt img to roi
            tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
            mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ < 0)

            if self.hpams['arch'] == 'autorf':
                # preprocess image and predict shapecode and texturecode
                # img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                shapecode = shapecode.detach().requires_grad_()
                texturecode = texturecode.detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')

            # set pose parameters
            rot_mat_vec = pred_pose[:3, :3].unsqueeze(0)
            trans_vec = pred_pose[:3, 3].unsqueeze(0).to(self.device).detach().requires_grad_()
            if self.hpams['euler_rot']:
                rot_vec = rot_trans.matrix_to_euler_angles(rot_mat_vec, 'XYZ').to(self.device).detach().requires_grad_()
            else:
                rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            # Optimization
            self.nopts = 0
            self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            est_cams = torch.zeros((1, 3, 4), dtype=torch.float32)
            while self.nopts < self.hpams['optimize']['num_opts']:
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                loss_per_img = []

                t2opt = trans_vec[0].unsqueeze(-1)
                if self.hpams['euler_rot']:
                    rot_mat2opt = rot_trans.euler_angles_to_matrix(rot_vec[0], 'XYZ')
                else:
                    rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[0])

                if self.hpams['optimize']['opt_obj_pose']:
                    rot_mat2opt = torch.transpose(rot_mat2opt, dim0=-2, dim1=-1)
                    t2opt = -rot_mat2opt @ t2opt

                cam2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                # render ray values and prepare target rays
                rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays(self.model, self.device,
                                                                                        tgt_img, mask_occ, cam2opt,
                                                                                        obj_diag, K, roi,
                                                                                        self.hpams['n_rays'],
                                                                                        self.hpams['n_samples'],
                                                                                        shapecode, texturecode,
                                                                                        self.hpams['shapenet_obj_cood'],
                                                                                        self.hpams['sym_aug'])
                # Compute losses
                # Critical to let rgb supervised on white background
                # loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                            torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # Occupancy loss
                loss_occ = torch.sum(
                    torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                       torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ

                if self.hpams['obj_sz_reg']:
                    sz_reg_samples = generate_obj_sz_reg_samples(obj_sz, obj_diag, self.hpams['shapenet_obj_cood'], tau=0.05)
                    loss_obj_sz = self.loss_obj_sz(sz_reg_samples, shapecode, texturecode)
                    loss = loss + self.hpams['loss_obj_sz_coef'] * loss_obj_sz

                loss.backward()
                loss_per_img.append(loss_rgb.detach().item())
                # Different roi sizes are dealt in save_image later
                gt_imgs.append(imgs[0, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                gt_masks_occ.append(masks_occ[0, roi[1]:roi[3], roi[0]:roi[2]])
                est_cams[0] = cam2opt.detach().cpu()
                errs_R, errs_T = calc_obj_pose_err(est_cams, tgt_cam.unsqueeze(0))
                if self.nopts == 0:
                    print('   Initial R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                    print('   Final R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                    self.log_eval_psnr(loss_per_img, batch_data['anntoken'])
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        # for num, cam_id in enumerate(cam_ids):
                        cam2opt = est_cams[0]
                        # render full image
                        generated_img = render_full_img(self.model, self.device, cam2opt, obj_sz, K, roi,
                                                        self.hpams['n_samples'], shapecode, texturecode,
                                                        self.hpams['shapenet_obj_cood'])
                        # mark pose error on the image
                        err_str = 'R err: {:.3f}, T err: {:.3f}'.format(errs_R[0], errs_T[0])
                        generated_img = cv2.putText(generated_img.cpu().numpy(), err_str, (5, 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, .3, (1, 0, 0))
                        generated_imgs.append(torch.from_numpy(generated_img))
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, anntoken, self.nopts)

                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                                                               self.hpams['n_samples'], shapecode, texturecode,
                                                               self.hpams['shapenet_obj_cood'])
                            self.save_virtual_img(virtual_imgs, anntoken, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.optimized_ann_flag[anntoken] = 1
            self.log_eval_pose(est_cams, tgt_cam.unsqueeze(0), batch_data['anntoken'])
            self.save_opts_w_pose(batch_idx)

    def optimize_objs_multi_anns(self, save_img=True):
        """
            optimize multiple annotations for the same instance in a singe iteration
        """

        instokens = self.nusc_dataset.anntokens_per_ins.keys()

        # Per object
        for obj_idx, instoken in enumerate(instokens):
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}')

            batch_data = self.nusc_dataset.get_ins_samples(instoken)

            tgt_imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            tgt_poses = batch_data['cam_poses']
            anntokens = batch_data['anntokens']

            if len(tgt_imgs) == 0:
                continue

            print(f'    num views: {tgt_imgs.shape[0]}')
            H, W = tgt_imgs.shape[1:3]
            rois[..., 0:2] -= self.hpams['roi_margin']
            rois[..., 2:4] += self.hpams['roi_margin']
            rois[..., 0:2] = torch.maximum(rois[..., 0:2], torch.as_tensor(0))
            rois[..., 2] = torch.minimum(rois[..., 2], torch.as_tensor(W - 1))
            rois[..., 3] = torch.minimum(rois[..., 3], torch.as_tensor(H - 1))

            if self.hpams['arch'] == 'autorf':
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
                    # img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                    img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                    shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                    shapecode_list.append(shapecode)
                    texturecode_list.append(texturecode)
                shapecode = torch.mean(torch.cat(shapecode_list), dim=0, keepdim=True).detach().requires_grad_()
                texturecode = torch.mean(torch.cat(texturecode_list), dim=0, keepdim=True).detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')

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
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays(self.model, self.device,
                                                                                            tgt_img, mask_occ, tgt_pose,
                                                                                            obj_diag, K, roi,
                                                                                            self.hpams['n_rays'],
                                                                                            self.hpams['n_samples'],
                                                                                            shapecode, texturecode,
                                                                                            self.hpams['shapenet_obj_cood'],
                                                                                            self.hpams['sym_aug'])
                    # Compute losses
                    # loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                    loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                                torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # Occupancy loss
                    loss_occ = torch.sum(
                        torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                           torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                    loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                    loss.backward()
                    loss_per_img.append(loss_rgb.detach().item())

                    # Different roi sizes are dealt in save_image later
                    gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])
                    self.optimized_ann_flag[anntokens[num]] = 1
                if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                    self.log_eval_psnr(loss_per_img, anntokens)
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        for num in range(0, tgt_imgs.shape[0]):
                            tgt_pose, roi, K = tgt_poses[num], rois[num], cam_intrinsics[num]
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                            # render full image
                            generated_img = render_full_img(self.model, self.device, tgt_pose, obj_sz, K, roi,
                                                            self.hpams['n_samples'], shapecode, texturecode,
                                                            self.hpams['shapenet_obj_cood'])
                            generated_imgs.append(generated_img)
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, instoken, self.nopts)

                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[0])['size']
                            virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                                                               self.hpams['n_samples'], shapecode, texturecode,
                                                               self.hpams['shapenet_obj_cood'])
                            self.save_virtual_img(virtual_imgs, instoken, self.nopts)

                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers(shapecode, texturecode)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.save_opts(obj_idx)

    def optimize_objs_multi_anns_w_pose(self, save_img=True):
        """
            optimize multiple annotations for the same instance in a singe iteration
        """

        instokens = self.nusc_dataset.anntokens_per_ins.keys()
        # Per object
        for obj_idx, instoken in enumerate(instokens):
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}')

            batch_data = self.nusc_dataset.get_ins_samples(instoken)

            tgt_imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            cam_poses = batch_data['cam_poses']
            cam_poses_w_err = batch_data['cam_poses_w_err']
            obj_poses_w_err = batch_data['obj_poses_w_err']
            anntokens = batch_data['anntokens']

            if self.hpams['optimize']['opt_obj_pose']:
                pred_poses = obj_poses_w_err
            else:
                pred_poses = cam_poses_w_err

            if len(tgt_imgs) == 0:
                continue

            print(f'    num views: {tgt_imgs.shape[0]}')
            H, W = tgt_imgs.shape[1:3]
            rois[..., 0:2] -= self.hpams['roi_margin']
            rois[..., 2:4] += self.hpams['roi_margin']
            rois[..., 0:2] = torch.maximum(rois[..., 0:2], torch.as_tensor(0))
            rois[..., 2] = torch.minimum(rois[..., 2], torch.as_tensor(W - 1))
            rois[..., 3] = torch.minimum(rois[..., 3], torch.as_tensor(H - 1))

            if self.hpams['arch'] == 'autorf':
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
                    # img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                    img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                    shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                    shapecode_list.append(shapecode)
                    texturecode_list.append(texturecode)
                shapecode = torch.mean(torch.cat(shapecode_list), dim=0, keepdim=True).detach().requires_grad_()
                texturecode = torch.mean(torch.cat(texturecode_list), dim=0, keepdim=True).detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')

            # set pose parameters
            rot_mat_vec = pred_poses[:, :3, :3]
            trans_vec = pred_poses[:, :3, 3].to(self.device).detach().requires_grad_()
            if self.hpams['euler_rot']:
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
                    tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], cam_poses[num], masks_occ[num], rois[num], \
                                                          cam_intrinsics[num]

                    t2opt = trans_vec[num].unsqueeze(-1)
                    if self.hpams['euler_rot']:
                        rot_mat2opt = rot_trans.euler_angles_to_matrix(rot_vec[num], 'XYZ')
                    else:
                        rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[num])

                    if self.hpams['optimize']['opt_obj_pose']:
                        rot_mat2opt = torch.transpose(rot_mat2opt, dim0=-2, dim1=-1)
                        t2opt = -rot_mat2opt @ t2opt

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
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays(self.model, self.device,
                                                                                            tgt_img, mask_occ, pose2opt,
                                                                                            obj_diag, K, roi,
                                                                                            self.hpams['n_rays'],
                                                                                            self.hpams['n_samples'],
                                                                                            shapecode, texturecode,
                                                                                            self.hpams['shapenet_obj_cood'],
                                                                                            self.hpams['sym_aug'])

                    # Compute losses
                    # loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                    loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                                torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # Occupancy loss
                    loss_occ = torch.sum(
                        torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                           torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                    loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                    loss.backward()
                    loss_per_img.append(loss_rgb.detach().item())

                    # Different roi sizes are dealt in save_image later
                    gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])
                    self.optimized_ann_flag[anntokens[num]] = 1
                    est_poses[num] = pose2opt.detach().cpu()
                errs_R, errs_T = calc_obj_pose_err(est_poses, cam_poses)
                if self.nopts == 0:
                    print('    Initial R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                    print('    Final R err: {:.3f}, T err: {:.3f}'.format(errs_R.mean(), errs_T.mean()))
                    self.log_eval_psnr(loss_per_img, anntokens)
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if save_img or self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                    # generate the full images
                    generated_imgs = []

                    with torch.no_grad():
                        for num in range(0, tgt_imgs.shape[0]):
                            tgt_pose, roi, K = cam_poses[num], rois[num], cam_intrinsics[num]
                            pose2opt = est_poses[num]
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                            # render full image
                            generated_img = render_full_img(self.model, self.device, pose2opt, obj_sz, K, roi,
                                                            self.hpams['n_samples'], shapecode, texturecode,
                                                            self.hpams['shapenet_obj_cood'])
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
                            virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                                                               self.hpams['n_samples'], shapecode, texturecode,
                                                               self.hpams['shapenet_obj_cood'])
                            self.save_virtual_img(virtual_imgs, instoken, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.log_eval_pose(est_poses, cam_poses, anntokens)
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

    def loss_sym(self, xyz, viewdir, sigmas, shapecode, texturecode):
        xyz_sym = torch.clone(xyz)
        viewdir_sym = torch.clone(viewdir)
        if self.hpams['shapenet_obj_cood']:
            xyz_sym[:, :, 0] *= (-1)
            viewdir_sym[:, :, 0] *= (-1)
        else:
            xyz_sym[:, :, 1] *= (-1)
            viewdir_sym[:, :, 1] *= (-1)
        sigmas_sym, rgbs_sym = self.model(xyz_sym.to(self.device),
                                          viewdir_sym.to(self.device),
                                          shapecode, texturecode)
        loss_sym = torch.mean((sigmas - sigmas_sym) ** 2)
        return loss_sym

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

    def save_img(self, generated_imgs, gt_imgs, masks_occ, obj_id, instance_num):
        # H, W = gt_imgs[0].shape[:2]
        W_tgt = np.min([gt_img.shape[1] for gt_img in gt_imgs])

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
        imageio.imwrite(os.path.join(save_img_dir, 'opt' + '{:03d}'.format(instance_num) + '.png'), ret)

    def save_virtual_img(self, imgs, obj_id, instance_num=None):
        H, W = imgs[0].shape[:2]

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

    def log_compute_ssim(self, generated_imgs, gt_imgs, ann_tokens):
        # ATTENTION: preparing all generated_imgs is time-consuming for evaluation purpose
        for i, ann_token in enumerate(ann_tokens):
            generated_img_np = generated_imgs[i].detach().cpu().numpy()
            gt_img_np = gt_imgs[i].detach().cpu().numpy()
            ssim = compute_ssim(generated_img_np, gt_img_np, multichannel=True)
            if self.ssim_eval.get(ann_token) is None:
                self.ssim_eval[ann_token] = [ssim]
            else:
                self.ssim_eval[ann_token].append(ssim)

    def log_eval_psnr(self, loss_per_img, ann_tokens):
        # ATTENTION: the loss_per_img including BG can make the value lower than reported in paper
        for i, ann_token in enumerate(ann_tokens):
            psnr = -10 * np.log(loss_per_img[i]) / np.log(10)
            if self.psnr_eval.get(ann_token) is None:
                self.psnr_eval[ann_token] = [psnr]
            else:
                self.psnr_eval[ann_token].append(psnr)

    def log_eval_pose(self, est_poses, tgt_poses, ann_tokens):
        # convert back to object pose to evaluate
        est_R = est_poses[:, :3, :3].transpose(-1, -2)
        est_T = -torch.matmul(est_R, est_poses[:, :3, 3:]).squeeze(-1)
        tgt_R = tgt_poses[:, :3, :3].transpose(-1, -2)
        tgt_T = -torch.matmul(tgt_R, tgt_poses[:, :3, 3:]).squeeze(-1)

        err_R = rot_dist(est_R, tgt_R)
        err_T = torch.sqrt(torch.sum((est_T - tgt_T)**2, dim=-1))

        for i, ann_token in enumerate(ann_tokens):
            self.R_eval[ann_token] = err_R[i]
            self.T_eval[ann_token] = err_T[i]

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
        if self.hpams['arch'] == 'autorf':
            self.model = AutoRF(**self.hpams['net_hyperparams']).to(self.device)
        elif self.hpams['arch'] == 'codenerf':
            self.model = CodeNeRF(**self.hpams['net_hyperparams']).to(self.device)
        else:
            print('ERROR: No valid network architecture is declared in config file!')

    def load_model(self):
        saved_dir = self.hpams['model_dir']
        saved_path = os.path.join(saved_dir, 'models.pth')
        saved_data = torch.load(saved_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)
        # also load in mean codes for codenerf
        if self.hpams['arch'] == 'codenerf':
            # mean shape should only consider those optimized codes when some of those are not touched
            if 'optimized_idx' in saved_data.keys():
                optimized_idx = saved_data['optimized_idx'].numpy()
                self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'][optimized_idx > 0],
                                             dim=0).reshape(1, -1)
                self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'][optimized_idx > 0],
                                               dim=0).reshape(1, -1)
            else:
                self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'], dim=0).reshape(1, -1)
                self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'], dim=0).reshape(1, -1)

    def make_save_img_dir(self):
        save_dir_tmp = self.hpams['model_dir'] + '/test' + self.save_postfix
        num = 2
        while os.path.isdir(save_dir_tmp):
            save_dir_tmp = self.hpams['model_dir'] + '/test' + self.save_postfix + '_' + str(num)
            num += 1

        os.makedirs(save_dir_tmp)
        self.save_dir = save_dir_tmp
