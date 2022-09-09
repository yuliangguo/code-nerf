import random

import numpy as np
import torchvision
import torch
import torch.nn as nn
import json
from utils import get_rays_nuscenes, sample_from_rays, volume_rendering, volume_rendering2, image_float_to_uint8, rot_dist
from skimage.metrics import structural_similarity as compute_ssim
from model import CodeNeRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import imageio
import time
import pytorch3d.transforms.rotation_conversions as rot_trans


class OptimizerNuScenes:
    def __init__(self, model_dir, gpu, nusc_dataset, jsonfile='srncar.json',
                 batch_size=2048, num_opts=200, num_cams_per_sample=1,
                 max_rot_pert=0, max_t_pert=0, save_postfix='_nuscenes', num_workers=0, shuffle=False):
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
        self.load_model_codes(model_dir)
        self.nusc_dataset = nusc_dataset
        self.dataloader = DataLoader(self.nusc_dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle)
        print('we are going to save at ', self.save_dir)
        #self.saved_dir = saved_dir
        self.B = batch_size
        self.num_opts = num_opts
        self.num_cams_per_sample = num_cams_per_sample
        self.nviews = str(num_cams_per_sample)
        self.psnr_eval = {}
        self.psnr_opt = {}
        self.ssim_eval = {}
        self.R_eval = {}
        self.T_eval = {}
        self.max_rot_pert = max_rot_pert
        self.max_t_pert = max_t_pert
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
                self.optimized_shapecodes[instoken] = self.mean_shape.detach().clone()
                self.optimized_texturecodes[instoken] = self.mean_texture.detach().clone()
                self.optimized_ins_flag[instoken] = 0

    def optimize_objs(self, lr=1e-2, lr_half_interval=50, save_img=True, roi_margin=5):
        """
            Optimize on each annotation frame independently
        """

        self.lr, self.lr_half_interval, iters = lr, lr_half_interval, 0
        # cam_ids = torch.tensor(cam_ids)
        cam_ids = [ii for ii in range(0, self.num_cams_per_sample)]
        # Per object
        for obj_dix, d in enumerate(self.dataloader):
            print(f'num obj: {obj_dix}/{len(self.dataloader)}')
            # focal, H, W, imgs, poses, obj_idx = d
            imgs, masks_occ, rois, camera_intrinsics, poses, valid_flags, instoken, anntoken = d
            tgt_imgs, tgt_poses, masks_occ, rois, camera_intrinsics, valid_flags = \
                imgs[0, cam_ids], poses[0, cam_ids], masks_occ[0, cam_ids], \
                rois[0, cam_ids], camera_intrinsics[0, cam_ids], valid_flags[0, cam_ids]
            if np.sum(valid_flags.numpy()) == 0:
                continue

            instoken, anntoken = instoken[0], anntoken[0]
            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
            rois[:, 0:2] -= roi_margin
            rois[:, 2:4] += roi_margin

            self.nopts, self.lr_half_interval = 0, lr_half_interval
            shapecode = self.optimized_shapecodes[instoken].to(self.device).detach().requires_grad_()
            texturecode = self.optimized_texturecodes[instoken].to(self.device).detach().requires_grad_()

            # First Optimize
            self.set_optimizers(shapecode, texturecode)
            while self.nopts < self.num_opts:
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                loss_per_img = []
                for num, cam_id in enumerate(cam_ids):
                    tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], tgt_poses[num], masks_occ[num], rois[num], camera_intrinsics[num]

                    # near and far sample range need to be adaptively calculated
                    near = np.linalg.norm(tgt_pose[:, -1]) - obj_diag / 2
                    far = np.linalg.norm(tgt_pose[:, -1]) + obj_diag / 2

                    rays_o, viewdir = get_rays_nuscenes(K, tgt_pose, roi)

                    # For different sized roi, extract a random subset of pixels with fixed batch size
                    n_rays = np.minimum(rays_o.shape[0], self.B)
                    random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
                    rays_o = rays_o[random_ray_ids]
                    viewdir = viewdir[random_ray_ids]
                    # The random selection should be within the roi pixels
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]].reshape(-1, 3)
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].reshape(-1, 1)
                    tgt_img = tgt_img[random_ray_ids].to(self.device)
                    mask_occ = mask_occ[random_ray_ids].to(self.device)
                    mask_rgb = torch.clone(mask_occ)
                    mask_rgb[mask_rgb < 0] = 0

                    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far,
                                                            self.hpams['N_samples'])
                    # TODO: how the object space is normalized and transferred shape net
                    xyz /= obj_diag
                    xyz = xyz[:, :, [1, 0, 2]]
                    viewdir = viewdir[:, :, [1, 0, 2]]

                    sigmas, rgbs = self.model(xyz.to(self.device),
                                              viewdir.to(self.device),
                                              shapecode, texturecode)
                    rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(self.device))
                    loss_rgb = torch.sum((rgb_rays - tgt_img) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                    # Occupancy loss is essential, the BG portion adjust the nerf as well
                    loss_occ = - torch.sum(torch.log(mask_occ * (0.5 - acc_trans_rays) + 0.5 + 1e-9) * torch.abs(mask_occ)) / (torch.sum(torch.abs(mask_occ))+1e-9)
                    reg_loss = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                    loss_reg = self.hpams['loss_reg_coef'] * torch.mean(reg_loss)
                    loss = loss_rgb + loss_occ * 1e-5 + loss_reg
                    loss.backward()
                    loss_per_img.append(loss_rgb.item())
                    # Different roi sizes are dealt in save_image later
                    gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])

                self.opts.step()
                self.log_opt_psnr_time(np.mean(loss_per_img), time.time() - t1, self.nopts + self.num_opts * obj_dix,
                                       obj_dix)
                self.log_regloss(reg_loss.item(), self.nopts, obj_dix)

                # Just use the cropped region instead to save computation on the visualization
                if save_img or self.nopts == 0 or self.nopts == (self.num_opts-1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        for num, cam_id in enumerate(cam_ids):
                            tgt_pose, roi, K = tgt_poses[num], rois[num], camera_intrinsics[num]

                            # near and far sample range need to be adaptively calculated
                            near = np.linalg.norm(tgt_pose[:, -1]) - obj_diag / 2
                            far = np.linalg.norm(tgt_pose[:, -1]) + obj_diag / 2

                            rays_o, viewdir = get_rays_nuscenes(K, tgt_pose, roi)
                            xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far,
                                                                    self.hpams['N_samples'])
                            # TODO: how the object space is normalized and transferred shape net
                            xyz /= obj_diag
                            xyz = xyz[:, :, [1, 0, 2]]
                            viewdir = viewdir[:, :, [1, 0, 2]]

                            generated_img = []
                            sample_step = np.maximum(roi[2]-roi[0], roi[3]-roi[1])
                            for i in range(0, xyz.shape[0], sample_step):
                                sigmas, rgbs = self.model(xyz[i:i + sample_step].to(self.device),
                                                          viewdir[i:i + sample_step].to(self.device),
                                                          shapecode, texturecode)
                                rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                                generated_img.append(rgb_rays)
                            generated_imgs.append(torch.cat(generated_img).reshape(roi[3]-roi[1], roi[2]-roi[0], 3))
                    self.save_img(generated_imgs, gt_imgs, gt_masks_occ, anntoken, self.nopts)

                self.nopts += 1
                if self.nopts % lr_half_interval == 0:
                    self.set_optimizers(shapecode, texturecode)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.optimized_ann_flag[anntoken] = 1
            self.save_opts(obj_dix)

    def optimize_objs_multi_anns(self, lr=1e-2, lr_half_interval=50, save_img=True, roi_margin=5):
        """
            optimize multiple annotations for the same instance in a singe iteration
        """

        self.lr, self.lr_half_interval, iters = lr, lr_half_interval, 0
        instokens = self.nusc_dataset.ins_ann_tokens.keys()

        # Per object
        for obj_idx, instoken in enumerate(instokens):
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}')

            tgt_imgs, masks_occ, rois, camera_intrinsics, tgt_poses, anntokens = self.nusc_dataset.get_ins_samples(instoken)

            if tgt_imgs is None:
                continue

            # if tgt_imgs is None or tgt_imgs.shape[0] < 2:
            #     continue

            print(f'    num views: {tgt_imgs.shape[0]}')

            rois[..., 0:2] -= roi_margin
            rois[..., 2:4] += roi_margin

            self.nopts, self.lr_half_interval = 0, lr_half_interval
            shapecode = self.optimized_shapecodes[instoken].to(self.device).detach().requires_grad_()
            texturecode = self.optimized_texturecodes[instoken].to(self.device).detach().requires_grad_()

            # Optimize
            self.set_optimizers(shapecode, texturecode)
            while self.nopts < self.num_opts:
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                loss_per_img = []
                for num in range(0, tgt_imgs.shape[0]):
                    tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], tgt_poses[num], masks_occ[num], rois[num], camera_intrinsics[num]

                    obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

                    # near and far sample range need to be adaptively calculated
                    near = np.linalg.norm(tgt_pose[:, -1]) - obj_diag / 2
                    far = np.linalg.norm(tgt_pose[:, -1]) + obj_diag / 2

                    rays_o, viewdir = get_rays_nuscenes(K, tgt_pose, roi)

                    # For different sized roi, extract a random subset of pixels with fixed batch size
                    n_rays = np.minimum(rays_o.shape[0], self.B)
                    random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
                    rays_o = rays_o[random_ray_ids]
                    viewdir = viewdir[random_ray_ids]
                    # The random selection should be within the roi pixels
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]].reshape(-1, 3)
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].reshape(-1, 1)
                    tgt_img = tgt_img[random_ray_ids].to(self.device)
                    mask_occ = mask_occ[random_ray_ids].to(self.device)
                    mask_rgb = torch.clone(mask_occ)
                    mask_rgb[mask_rgb < 0] = 0

                    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far,
                                                            self.hpams['N_samples'])
                    # TODO: how the object space is normalized and transferred shape net
                    xyz /= obj_diag
                    xyz = xyz[:, :, [1, 0, 2]]
                    viewdir = viewdir[:, :, [1, 0, 2]]

                    sigmas, rgbs = self.model(xyz.to(self.device),
                                              viewdir.to(self.device),
                                              shapecode, texturecode)
                    rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(self.device))
                    loss_rgb = torch.sum((rgb_rays - tgt_img) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                    # Occupancy loss is essential, the BG portion adjust the nerf as well
                    loss_occ = - torch.sum(torch.log(mask_occ * (0.5 - acc_trans_rays) + 0.5 + 1e-9) * torch.abs(mask_occ)) / (torch.sum(torch.abs(mask_occ))+1e-9)
                    reg_loss = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                    loss_reg = self.hpams['loss_reg_coef'] * torch.mean(reg_loss)
                    loss = loss_rgb + loss_occ * 1e-5 + loss_reg
                    loss.backward()
                    loss_per_img.append(loss_rgb.item())

                    # Different roi sizes are dealt in save_image later
                    gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])
                    self.optimized_ann_flag[anntokens[num]] = 1

                self.opts.step()
                self.log_opt_psnr_time(np.mean(loss_per_img), time.time() - t1, self.nopts + self.num_opts * obj_idx,
                                       obj_idx)
                self.log_regloss(reg_loss.item(), self.nopts, obj_idx)

                # Just render the cropped region instead to save computation on the visualization
                if save_img or self.nopts == 0 or self.nopts == (self.num_opts-1):
                    # generate the full images
                    generated_imgs = []
                    with torch.no_grad():
                        for num in range(0, tgt_imgs.shape[0]):
                            tgt_pose, roi, K = tgt_poses[num], rois[num], camera_intrinsics[num]

                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

                            # near and far sample range need to be adaptively calculated
                            near = np.linalg.norm(tgt_pose[:, -1]) - obj_diag / 2
                            far = np.linalg.norm(tgt_pose[:, -1]) + obj_diag / 2

                            rays_o, viewdir = get_rays_nuscenes(K, tgt_pose, roi)
                            xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far,
                                                                    self.hpams['N_samples'])
                            # TODO: how the object space is normalized and transferred shape net
                            xyz /= obj_diag
                            xyz = xyz[:, :, [1, 0, 2]]
                            viewdir = viewdir[:, :, [1, 0, 2]]

                            generated_img = []
                            sample_step = np.maximum(roi[2]-roi[0], roi[3]-roi[1])
                            for i in range(0, xyz.shape[0], sample_step):
                                sigmas, rgbs = self.model(xyz[i:i + sample_step].to(self.device),
                                                          viewdir[i:i + sample_step].to(self.device),
                                                          shapecode, texturecode)
                                rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                                generated_img.append(rgb_rays)
                            generated_imgs.append(torch.cat(generated_img).reshape(roi[3]-roi[1], roi[2]-roi[0], 3))
                    self.save_img(generated_imgs, gt_imgs, gt_masks_occ, instoken, self.nopts)

                self.nopts += 1
                if self.nopts % lr_half_interval == 0:
                    self.set_optimizers(shapecode, texturecode)

            # Save the optimized codes
            self.optimized_shapecodes[instoken] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken] = texturecode.detach().cpu()
            self.optimized_ins_flag[instoken] = 1
            self.save_opts(obj_idx)

    def save_opts(self, num_obj):
        saved_dict = {
            'num_obj': num_obj,
            'optimized_shapecodes' : self.optimized_shapecodes,
            'optimized_texturecodes': self.optimized_texturecodes,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes.pth'))
        print('We finished the optimization of ' + str(num_obj))

    def save_opts_w_pose(self, num_obj):
        saved_dict = {
            'num_obj' : num_obj,
            'optimized_shapecodes': self.optimized_shapecodes,
            'optimized_texturecodes': self.optimized_texturecodes,
            'optimized_poses': self.optimized_poses,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval,
            'R_eval': self.R_eval,
            'T_eval': self.T_eval
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes+poses.pth'))
        print('We finished the optimization of ' + str(num_obj))

    def save_img(self, generated_imgs, gt_imgs, masks_occ, obj_id, instance_num):
        # H, W = gt_imgs[0].shape[:2]
        W_tgt = np.min([gt_img.shape[1] for gt_img in gt_imgs])
        # nviews = len(gt_imgs)

        if len(gt_imgs) > 1:
            # Align the width of different-sized images
            generated_imgs = self.align_imgs_width(generated_imgs, W_tgt)
            masks_occ = self.align_imgs_width(masks_occ, W_tgt)
            gt_imgs = self.align_imgs_width(gt_imgs, W_tgt)

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
        imageio.imwrite(os.path.join(save_img_dir, 'opt' + self.nviews + '_{:03d}'.format(instance_num) + '.png'), ret)

    def align_imgs_width(self, imgs, W, max_view=4):
        """
            imgs: a list of tensors
        """
        out_imgs = []
        if len(imgs) > max_view:
            step = len(imgs) // max_view
        else:
            step = 1

        for id in range(0, len(imgs), step):
            img = imgs[id]
            H_i, W_i = img.shape[:2]
            H_out = int(float(H_i) * W / W_i)
            # out_imgs.append(Image.fromarray(img.detach().cpu().numpy()).resize((W, H_out)))
            img = img.reshape((1, H_i, W_i, -1))
            img = img.permute((0, 3, 1, 2))
            img = torchvision.transforms.Resize((H_out, W))(img)
            img = img.permute((0, 2, 3, 1))
            out_imgs.append(img.squeeze())
            if len(out_imgs) == max_view:
                break
        return out_imgs

    def log_compute_ssim(self, generated_img, gt_img, niters, obj_idx):
        generated_img_np = generated_img.detach().cpu().numpy()
        gt_img_np = gt_img.detach().cpu().numpy()
        ssim = compute_ssim(generated_img_np, gt_img_np, multichannel=True)
        # if niters == 0:
        if self.ssim_eval.get(obj_idx) is None:
            self.ssim_eval[obj_idx] = [ssim]
        else:
            self.ssim_eval[obj_idx].append(ssim)

    def log_eval_psnr(self, loss_per_img, niters, obj_idx):
        psnr = -10 * np.log(loss_per_img) / np.log(10)
        # if niters == 0:
        if self.psnr_eval.get(obj_idx) is None:
            self.psnr_eval[obj_idx] = [psnr]
        else:
            self.psnr_eval[obj_idx].append(psnr)

    def log_eval_pose(self, est_poses, tgt_poses, obj_idx):
        est_R = est_poses[:, :3, :3]
        est_T = est_poses[:, :3, 3]
        tgt_R = tgt_poses[:, :3, :3]
        tgt_T = tgt_poses[:, :3, 3]

        err_R = rot_dist(est_R, tgt_R)
        err_T = torch.sqrt(torch.sum((est_T - tgt_T)**2, dim=-1))
        if self.R_eval.get(obj_idx) is None:
            self.R_eval[obj_idx] = [err_R]
            self.T_eval[obj_idx] = [err_T]
        else:
            self.R_eval[obj_idx].append(err_R)
            self.T_eval[obj_idx].append(err_T)

    def log_opt_psnr_time(self, loss_per_img, time_spent, niters, obj_idx):
        psnr = -10*np.log(loss_per_img) / np.log(10)
        self.writer.add_scalar('psnr_opt/' + self.nviews, psnr, niters, obj_idx)
        self.writer.add_scalar('time_opt/' + self.nviews, time_spent, niters, obj_idx)

    def log_regloss(self, loss_reg, niters, obj_idx):
        self.writer.add_scalar('reg/' + self.nviews, loss_reg, niters, obj_idx)

    def set_optimizers(self, shapecode, texturecode):
        lr = self.get_learning_rate()
        #print(lr)
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': lr},
            {'params': texturecode, 'lr':lr}
        ])

    def set_optimizers_w_pose(self, shapecode, texturecode, poses):
        lr = self.get_learning_rate()
        #print(lr)
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': lr},
            {'params': texturecode, 'lr': lr},
            {'params': poses, 'lr': lr},
        ])

    def set_optimizers_w_euler_poses(self, shapecode, texturecode, euler_angles, trans):
        lr = self.get_learning_rate()
        #print(lr)
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': lr},
            {'params': texturecode, 'lr': lr},
            {'params': euler_angles, 'lr': lr},
            {'params': trans, 'lr': lr}
        ])

    def get_learning_rate(self):
        opt_values = self.nopts // self.lr_half_interval
        lr = self.lr * 2**(-opt_values)
        return lr

    def make_model(self):
        self.model = CodeNeRF(**self.hpams['net_hyperparams']).to(self.device)

    def load_model_codes(self, saved_dir):
        saved_path = os.path.join('exps', saved_dir, 'models.pth')
        saved_data = torch.load(saved_path, map_location = torch.device('cpu'))
        self.make_save_img_dir(os.path.join('exps', saved_dir, 'test'))
        self.make_writer(saved_dir)
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)
        self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'], dim=0).reshape(1,-1)
        self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'], dim=0).reshape(1,-1)

    def make_writer(self, saved_dir):
        self.writer = SummaryWriter(os.path.join('exps', saved_dir, 'test', 'runs'))

    def make_save_img_dir(self, save_dir):
        save_dir_tmp = save_dir + self.save_postfix
        num = 2
        while os.path.isdir(save_dir_tmp):
            save_dir_tmp = save_dir + self.save_postfix + '_' + str(num)
            num += 1

        os.makedirs(save_dir_tmp)
        self.save_dir = save_dir_tmp
