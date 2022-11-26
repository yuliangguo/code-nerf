import random

import numpy as np
import torchvision
import torch
import torch.nn as nn
import json
from utils import get_rays_nuscenes, sample_from_rays, volume_rendering, volume_rendering2, image_float_to_uint8
from model_autorf import AutoRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
import os
import imageio
import time
import math


class TrainerAutoRFNuScenes:
    def __init__(self, save_dir, gpu, nusc_dataset, pretrained_model_dir=None, resume_from_epoch=None, jsonfile='srncar.json', batch_size=2,
                 num_workers=0, shuffle=False, check_iter=1000):
        """
        :param pretrained_model_dir: the directory of pre-trained model
        :param gpu: which GPU we would use
        :param jsonfile: where the hyper-parameters are saved
        """
        super().__init__()
        # Read Hyperparameters
        hpampath = os.path.join('jsonfiles', jsonfile)
        with open(hpampath, 'r') as f:
            self.hpams = json.load(f)
        self.device = torch.device('cuda:' + str(gpu))
        self.nusc_dataset = nusc_dataset
        self.dataloader = DataLoader(self.nusc_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        self.batch_size = batch_size
        self.niter, self.nepoch = 0, 0
        self.check_iter = check_iter

        self.make_savedir(save_dir)
        print('we are going to save at ', self.save_dir)

        # initialize model
        self.make_model()
        if pretrained_model_dir is not None:
            self.load_pretrained_model(pretrained_model_dir)

        # Load from epoch requires initialization before
        if resume_from_epoch is not None:
            self.resume_from_epoch(save_dir, resume_from_epoch)

    def training(self, epochs):
        # initialize the losses here to avoid memory leak between epochs
        self.losses_rgb = []
        self.losses_occ = []
        # self.losses_reg = []
        self.loss_total = torch.zeros(1).to(self.device)
        self.set_optimizers()
        self.opts.zero_grad()
        self.t1 = time.time()

        while self.nepoch < epochs:
            print(f'epoch: {self.nepoch}')
            self.training_single_epoch()

            self.save_models(epoch=self.nepoch)
            self.nepoch += 1

    def training_single_epoch(self, sym_aug=False):
        """
            Optimize on each annotation frame independently
        """

        cam_id = 0
        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            for obj_idx, imgs in enumerate(batch_data['imgs']):

                masks_occ = batch_data['masks_occ'][obj_idx]
                rois = batch_data['rois'][obj_idx]
                cam_intrinsics = batch_data['cam_intrinsics'][obj_idx]
                cam_poses = batch_data['cam_poses'][obj_idx]
                valid_flags = batch_data['valid_flags'][obj_idx]
                instoken = batch_data['instoken'][obj_idx]
                anntoken = batch_data['anntoken'][obj_idx]

                # skip unqualified sample
                if np.sum(valid_flags.numpy()) == 0:
                    continue

                # print(f'epoch: {self.nepoch}, batch: {batch_idx}/{len(self.dataloader)}, obj: {obj_idx} is qualified')
                obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']
                obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
                H, W = imgs.shape[1:3]
                rois[:, 0:2] -= self.hpams['roi_margin']
                rois[:, 2:4] += self.hpams['roi_margin']
                rois[:, 0:2] = torch.maximum(rois[:, 0:2], torch.as_tensor(0))
                rois[:, 2] = torch.minimum(rois[:, 2], torch.as_tensor(W-1))
                rois[:, 3] = torch.minimum(rois[:, 3], torch.as_tensor(H-1))

                tgt_img, tgt_pose, mask_occ, roi, K = \
                    imgs[cam_id], cam_poses[cam_id], masks_occ[cam_id], rois[cam_id], cam_intrinsics[cam_id]

                # crop tgt img to roi
                tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                tgt_img = tgt_img * (mask_occ > 0)
                tgt_img = tgt_img + (mask_occ < 0)

                # preprocess image and predict shapecode and texturecode
                img_in = self.preprocess_img(tgt_img)
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))

                # render ray values and prepare target rays
                rgb_rays, depth_rays, acc_trans_rays, tgt_rays, occ_pixels = self.render_rays(tgt_img, mask_occ,
                                                                                            tgt_pose, obj_diag, K,
                                                                                            roi, shapecode,
                                                                                            texturecode,
                                                                                            shapenet_obj_cood=True,
                                                                                            sym_aug=sym_aug)

                # loss_rgb = torch.sum((rgb_rays - tgt_pixels) ** 2 * mask_rgb) / (torch.sum(mask_rgb)+1e-9)
                loss_rgb = torch.sum((rgb_rays - tgt_rays) ** 2 * torch.abs(occ_pixels)) / (
                            torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # Occupancy loss
                loss_occ = torch.sum(
                    torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                   torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                # self.loss_total += (loss_rgb + self.hpams['loss_occ_coef'] * loss_occ + self.hpams['loss_reg_coef'] * loss_reg)
                self.loss_total += loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                # self.loss_total += loss_rgb + self.hpams['loss_reg_coef'] * loss_reg
                # self.loss_total = loss_rgb

                self.losses_rgb.append(loss_rgb.detach().item())
                self.losses_occ.append(loss_occ.detach().item())
                # self.losses_reg.append(loss_reg.detach().item())

                if self.niter % self.check_iter == 0:
                    # Just use the cropped region instead to save computation on the visualization
                    with torch.no_grad():
                        # near and far sample range need to be adaptively calculated
                        near = np.linalg.norm(tgt_pose[:, -1]) - obj_diag / 2
                        far = np.linalg.norm(tgt_pose[:, -1]) + obj_diag / 2

                        rays_o, viewdir = get_rays_nuscenes(K, tgt_pose, roi)
                        xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, self.hpams['n_samples'])
                        # Nuscene to ShapeNet: rotate 90 degree around Z and normalize to (-1 1)
                        xyz /= obj_diag
                        xyz = xyz[:, :, [1, 0, 2]]
                        xyz[:, :, 0] *= (-1)
                        viewdir = viewdir[:, :, [1, 0, 2]]
                        viewdir[:, :, 0] *= -1

                        generated_img = []
                        sample_step = np.maximum(roi[2] - roi[0], roi[3] - roi[1])
                        for i in range(0, xyz.shape[0], sample_step):
                            sigmas, rgbs = self.model(xyz[i:i + sample_step].to(self.device),
                                                      viewdir[i:i + sample_step].to(self.device),
                                                      shapecode,
                                                      texturecode)
                            rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                            generated_img.append(rgb_rays)
                        generated_img = torch.cat(generated_img).reshape(roi[3] - roi[1], roi[2] - roi[0], 3)

                    gt_img = imgs[cam_id, roi[1]:roi[3], roi[0]:roi[2]]
                    gt_mask_occ = masks_occ[cam_id, roi[1]:roi[3], roi[0]:roi[2]]
                    self.log_img(generated_img, gt_img, gt_mask_occ, anntoken)

                if len(self.losses_rgb) == self.batch_size:
                    print(f'    optimize niter: {self.niter}, epoch: {self.nepoch}, batch: {batch_idx}/{len(self.dataloader)}')
                    # optimize when collected a batch of qualified samples
                    self.loss_total.backward()
                    self.opts.step()
                    self.log_losses(np.mean(self.losses_rgb), np.mean(self.losses_occ), self.loss_total.detach().item(), time.time() - self.t1)

                    # reset all the losses
                    self.opts.zero_grad()
                    self.t1 = time.time()
                    self.losses_rgb = []
                    self.losses_occ = []
                    # self.losses_reg = []
                    self.loss_total = torch.zeros(1).to(self.device)

                    # iterations are only counted after optimized an qualified batch
                    self.niter += 1

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

        # extract samples
        tgt_pixels = img.reshape(-1, 3)[random_ray_ids].to(self.device)
        mask_occ = mask_occ.reshape(-1, 1)[random_ray_ids].to(self.device)
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
        return rgb_rays, depth_rays, acc_trans_rays, tgt_pixels, mask_occ

    def preprocess_img(self, img):
        img = img.unsqueeze(0).permute((0, 3, 1, 2))
        _, _, im_h, im_w = img.shape
        if np.maximum(im_h, im_w) > self.hpams['max_img_sz']:
            ratio = self.hpams['max_img_sz'] / np.maximum(im_h, im_w)
            new_h = im_h * ratio
            new_w = im_w * ratio
            img = Resize((int(new_h), int(new_w)))(img)
        return img

    def log_losses(self, loss_rgb, loss_occ, loss_total, time_spent):
        psnr = -10 * np.log(loss_rgb) / np.log(10)
        self.writer.add_scalar('psnr/train', psnr, self.niter)
        self.writer.add_scalar('loss_rgb/train', loss_rgb, self.niter)
        self.writer.add_scalar('loss_occ/train', loss_occ, self.niter)
        # self.writer.add_scalar('loss_reg/train', loss_reg, self.niter)
        self.writer.add_scalar('loss_total/train', loss_total, self.niter)
        self.writer.add_scalar('time/train', time_spent, self.niter)

    def log_img(self, generated_img, gtimg, mask_occ, ann_token):
        H, W = generated_img.shape[:-1]
        ret = torch.zeros(H, 2 * W, 3)
        ret[:, :W, :] = generated_img
        ret[:, W:, :] = gtimg * 0.7 + mask_occ.unsqueeze(-1) * 0.3
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        self.writer.add_image('train_' + str(self.niter) + '_' + ann_token,
                              torch.from_numpy(ret).permute(2, 0, 1))

    def set_optimizers(self):
        lr1, lr2 = self.get_learning_rate()
        self.opts = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': lr1},
        ])

    def get_learning_rate(self):
        model_lr, latent_lr = self.hpams['lr_schedule'][0], self.hpams['lr_schedule'][1]
        num_model = self.niter // model_lr['interval']
        num_latent = self.niter // latent_lr['interval']
        lr1 = model_lr['lr'] * 2 ** (-num_model)
        lr2 = latent_lr['lr'] * 2 ** (-num_latent)
        return lr1, lr2

    def make_model(self):
        self.model = AutoRF(**self.hpams['net_hyperparams']).to(self.device)

    def load_pretrained_model(self, saved_model_file):
        saved_data = torch.load(saved_model_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)

    def make_savedir(self, save_dir):
        self.save_dir = os.path.join(f'exps_nuscenes_autorf', save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(os.path.join(self.save_dir, 'runs'))
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'runs'))
        hpampath = os.path.join(self.save_dir, 'hpam.json')
        with open(hpampath, 'w') as f:
            json.dump(self.hpams, f, indent=2)

    def save_models(self, iter=None, epoch=None):
        save_dict = {'model_params': self.model.state_dict(),
                     'niter': self.niter,
                     'nepoch': self.nepoch,
                     }
        if iter != None:
            torch.save(save_dict, os.path.join(self.save_dir, str(iter) + '.pth'))
        if epoch != None:
            torch.save(save_dict, os.path.join(self.save_dir, f'epoch_{epoch}.pth'))
        torch.save(save_dict, os.path.join(self.save_dir, 'models.pth'))

    def resume_from_epoch(self, saved_dir, epoch):
        print(f'Resume training from saved model at epoch {epoch}.')
        saved_path = os.path.join('exps_nuscenes_autorf', saved_dir, f'epoch_{epoch}.pth')
        saved_data = torch.load(saved_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)
        self.niter = saved_data['niter'] + 1
        self.nepoch = saved_data['nepoch'] + 1
