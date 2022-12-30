import numpy as np
import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize

from utils import image_float_to_uint8, render_full_img, prepare_pixel_samples, volume_rendering_batch, preprocess_img_square
from model_autorf import AutoRF
from model_codenerf import CodeNeRF


class ParallelModel(nn.Module):
    def __init__(self, model=None, hpams=None):
        super().__init__()
        self.model = model
        self.hpams = hpams
    
    def forward(self,
                img_in_batch,
                shapecode_batch,
                texturecode_batch,
                xyz_batch,
                viewdir_batch,
                z_vals_batch,
                rgb_tgt_batch,
                occ_pixels_batch):
        if self.hpams['arch'] == 'autorf':
            shapecode_batch, texturecode_batch = self.model.encode_img(img_in_batch)

        sigmas, rgbs = self.model(xyz_batch.flatten(0, 1),
                                  viewdir_batch.flatten(0, 1),
                                  shapecode_batch,
                                  texturecode_batch)

        b_size = img_in_batch.shape[0]
        n, s, _ = sigmas.shape
        rgb_rays, depth_rays, acc_trans_rays = volume_rendering_batch(sigmas.view(b_size, int(n/b_size), s, -1),
                                                                      rgbs.view(b_size, int(n/b_size), s, -1),
                                                                      z_vals_batch)

        loss_rgb = torch.sum((rgb_rays - rgb_tgt_batch) ** 2 * torch.abs(occ_pixels_batch), dim=[-2, -1])/(
                torch.sum(torch.abs(occ_pixels_batch), dim=[-2, -1]) + 1e-9)

        # Occupancy loss
        loss_occ = torch.sum(
            torch.exp(-occ_pixels_batch * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels_batch),
            dim=[-2, -1]) / (torch.sum(torch.abs(occ_pixels_batch), dim=[-2, -1]) + 1e-9)
        loss_reg = torch.norm(shapecode_batch, dim=-1) + torch.norm(texturecode_batch, dim=-1)
        loss_total = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
        return loss_rgb.mean(), loss_occ.mean(), loss_reg.mean(), loss_total.mean()


class TrainerNuScenes:
    def __init__(self, save_dir, gpus, nusc_dataset, pretrained_model_dir=None, resume_from_epoch=None, jsonfile='srncar.json', batch_size=2,
                 num_workers=0, shuffle=False, check_iter=1000):
        """

        """
        super().__init__()
        # Read Hyperparameters
        with open(jsonfile, 'r') as f:
            self.hpams = json.load(f)
        # the device to hold data
        self.device = torch.device('cuda:' + str(0))
        self.nusc_dataset = nusc_dataset
        self.dataloader = DataLoader(self.nusc_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        self.batch_size = batch_size
        self.niter, self.nepoch = 0, 0
        self.check_iter = check_iter

        self.make_savedir(save_dir)
        print('we are going to save at ', self.save_dir)

        # initialize model
        self.make_model()
        self.parallel_model = torch.nn.DataParallel(
            ParallelModel(self.model, self.hpams), device_ids=list(range(gpus)))

        self.mean_shape = None
        self.mean_texture = None
        if pretrained_model_dir is not None:
            self.load_pretrained_model(pretrained_model_dir)

        # initialize shapecode, texturecode (only for codenerf to use)
        self.shape_codes = None
        self.texture_codes = None
        self.instoken2idx = {}
        idx = 0
        for ii, instoken in enumerate(self.nusc_dataset.anntokens_per_ins.keys()):
            if instoken not in self.instoken2idx.keys():
                self.instoken2idx[instoken] = idx
                idx += 1
        self.optimized_idx = torch.zeros(len(self.instoken2idx.keys()))
        self.make_codes()

        # Load from epoch requires initialization before
        if resume_from_epoch is not None:
            self.resume_from_epoch(save_dir, resume_from_epoch)

    def train(self, epochs):
        self.set_optimizers()
        self.opts.zero_grad()
        self.t1 = time.time()
        
        # initialize the accumulaters here to avoid memory leak between epochs
        self.img_in_batch = []
        self.shapecode_batch = []
        self.texturecode_batch = []
        self.xyz_batch = []
        self.viewdir_batch = []
        self.z_vals_batch = []
        self.rgb_tgt_batch = []
        self.occ_pixels_batch = []
        self.iter_vis_cnt = 0

        while self.nepoch < epochs:
            print(f'epoch: {self.nepoch}')
            self.training_epoch()

            self.save_models(epoch=self.nepoch)
            self.nepoch += 1

    def training_epoch(self):
        """
            Optimize on each annotation frame independently
        """

        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            # TODO: current data loader only load one image data per batch, so the real batch needs manual accumulation
            for obj_idx, img in enumerate(batch_data['imgs']):
                tgt_img = img.clone()
                mask_occ = batch_data['masks_occ'][obj_idx].clone()
                roi = batch_data['rois'][obj_idx]
                K = batch_data['cam_intrinsics'][obj_idx]
                tgt_pose = batch_data['cam_poses'][obj_idx]
                instoken = batch_data['instoken'][obj_idx]
                anntoken = batch_data['anntoken'][obj_idx]

                # print(f'epoch: {self.nepoch}, batch: {batch_idx}/{len(self.dataloader)}, obj: {obj_idx} is qualified')
                obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']
                obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
                H, W = tgt_img.shape[0:2]
                roi[0:2] -= self.hpams['roi_margin']
                roi[2:4] += self.hpams['roi_margin']
                roi[0:2] = torch.maximum(roi[0:2], torch.as_tensor(0))
                roi[2] = torch.minimum(roi[2], torch.as_tensor(W-1))
                roi[3] = torch.minimum(roi[3], torch.as_tensor(H-1))

                # crop tgt img to roi
                tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                tgt_img = tgt_img * (mask_occ > 0)
                tgt_img = tgt_img + (mask_occ < 0)

                # Preprocess img for model inference (pad and resize to the same square size)
                img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                
                code_idx = self.instoken2idx[instoken]
                code_idx = torch.as_tensor(code_idx).to(self.device)
                shapecode = self.shape_codes(code_idx).unsqueeze(0)
                texturecode = self.texture_codes(code_idx).unsqueeze(0)
                self.optimized_idx[code_idx.item()] = 1
                    
                xyz, viewdir, z_vals, rgb_tgt, occ_pixels = prepare_pixel_samples(tgt_img, mask_occ, tgt_pose,
                                                                                  obj_diag, K, roi,
                                                                                  self.hpams['n_rays'],
                                                                                  self.hpams['n_samples'],
                                                                                  self.hpams['shapenet_obj_cood'],
                                                                                  self.hpams['sym_aug'])
                
                self.img_in_batch.append(img_in)
                self.shapecode_batch.append(shapecode)
                self.texturecode_batch.append(texturecode)
                self.xyz_batch.append(xyz.unsqueeze(0))
                self.viewdir_batch.append(viewdir.unsqueeze(0))
                self.z_vals_batch.append(z_vals.unsqueeze(0))
                self.rgb_tgt_batch.append(rgb_tgt.unsqueeze(0))
                self.occ_pixels_batch.append(occ_pixels.unsqueeze(0))

                if self.niter % self.check_iter == 0 and self.iter_vis_cnt < 4:
                    # Just use the cropped region instead to save computation on the visualization
                    with torch.no_grad():
                        generated_img = render_full_img(self.model, self.device, tgt_pose, obj_sz, K, roi,
                                                        self.hpams['n_samples'], shapecode, texturecode,
                                                        self.hpams['shapenet_obj_cood'])
                    self.log_img(generated_img, img[roi[1]: roi[3], roi[0]: roi[2]], mask_occ, anntoken)
                    self.iter_vis_cnt += 1

                # TODO: preprocess of the dataset to only keep the valid sample
                if len(self.img_in_batch) == self.batch_size:
                    print(f'    optimize niter: {self.niter}, epoch: {self.nepoch}, batch: {batch_idx}/{len(self.dataloader)}')
                    # optimize when collected a batch of qualified samples

                    self.img_in_batch = torch.cat(self.img_in_batch).to(self.device)
                    self.shapecode_batch = torch.cat(self.shapecode_batch).to(self.device)
                    self.texturecode_batch = torch.cat(self.texturecode_batch).to(self.device)
                    self.xyz_batch = torch.cat(self.xyz_batch).to(self.device)
                    self.viewdir_batch = torch.cat(self.viewdir_batch).to(self.device)
                    self.z_vals_batch = torch.cat(self.z_vals_batch).to(self.device)
                    self.rgb_tgt_batch = torch.cat(self.rgb_tgt_batch).to(self.device)
                    self.occ_pixels_batch = torch.cat(self.occ_pixels_batch).to(self.device)

                    # compute losses from parallel model
                    loss_rgb, loss_occ, loss_reg, loss_total = self.parallel_model(self.img_in_batch,
                                                                                   self.shapecode_batch,
                                                                                   self.texturecode_batch,
                                                                                   self.xyz_batch,
                                                                                   self.viewdir_batch,
                                                                                   self.z_vals_batch,
                                                                                   self.rgb_tgt_batch,
                                                                                   self.occ_pixels_batch)
                    # compute gradient from the mean loss over multiple gpus
                    loss_total.mean().backward()
                    self.opts.step()
                    self.log_losses(loss_rgb.mean().detach().item(), loss_occ.mean().detach().item(), 
                                    loss_reg.mean().detach().item(), loss_total.mean().detach().item(), time.time() - self.t1)

                    # reset all the losses
                    self.opts.zero_grad()
                    self.t1 = time.time()
                    
                    self.img_in_batch = []
                    self.shapecode_batch = []
                    self.texturecode_batch = []
                    self.xyz_batch = []
                    self.viewdir_batch = []
                    self.z_vals_batch = []
                    self.rgb_tgt_batch = []
                    self.occ_pixels_batch = []
                    self.iter_vis_cnt = 0

                    # iterations are only counted after optimized an qualified batch
                    self.niter += 1

    def log_losses(self, loss_rgb, loss_occ, loss_reg, loss_total, time_spent):
        psnr = -10 * np.log(loss_rgb) / np.log(10)
        self.writer.add_scalar('psnr/train', psnr, self.niter)
        self.writer.add_scalar('loss_rgb/train', loss_rgb, self.niter)
        self.writer.add_scalar('loss_occ/train', loss_occ, self.niter)
        self.writer.add_scalar('loss_reg/train', loss_reg, self.niter)
        self.writer.add_scalar('loss_total/train', loss_total, self.niter)
        self.writer.add_scalar('time/train', time_spent, self.niter)

    def log_img(self, generated_img, gtimg, mask_occ, ann_token):
        H, W = generated_img.shape[:-1]
        ret = torch.zeros(H, 2 * W, 3)
        ret[:, :W, :] = generated_img
        ret[:, W:, :] = gtimg * 0.7 + mask_occ * 0.3
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        self.writer.add_image('train_' + str(self.niter) + '_' + ann_token,
                              torch.from_numpy(ret).permute(2, 0, 1))

    def set_optimizers(self):
        lr1, lr2 = self.get_learning_rate()
        if self.hpams['arch'] == 'autorf':
            self.opts = torch.optim.AdamW([{'params': self.model.parameters(), 'lr': lr1}])
        elif self.hpams['arch'] == 'codenerf':
            self.opts = torch.optim.AdamW([
                {'params': self.model.parameters(), 'lr': lr1},
                {'params': self.shape_codes.parameters(), 'lr': lr2},
                {'params': self.texture_codes.parameters(), 'lr': lr2}
            ])
        else:
            print('ERROR: No valid network architecture is declared in config file!')

    def get_learning_rate(self):
        model_lr, latent_lr = self.hpams['lr_schedule'][0], self.hpams['lr_schedule'][1]
        num_model = self.niter // model_lr['interval']
        num_latent = self.niter // latent_lr['interval']
        lr1 = model_lr['lr'] * 2 ** (-num_model)
        lr2 = latent_lr['lr'] * 2 ** (-num_latent)
        return lr1, lr2

    def make_model(self):
        if self.hpams['arch'] == 'autorf':
            self.model = AutoRF(**self.hpams['net_hyperparams']).to(self.device)
        elif self.hpams['arch'] == 'codenerf':
            self.model = CodeNeRF(**self.hpams['net_hyperparams']).to(self.device)
        else:
            print('ERROR: No valid network architecture is declared in config file!')

    def make_codes(self):
        d = len(self.instoken2idx.keys())
        embdim = self.hpams['net_hyperparams']['latent_dim']
        self.shape_codes = nn.Embedding(d, embdim)
        self.texture_codes = nn.Embedding(d, embdim)
        if self.mean_shape is None:
            self.shape_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim / 2))
            self.texture_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim / 2))
        else:
            self.shape_codes.weight = nn.Parameter(self.mean_shape.repeat(d, 1))
            self.texture_codes.weight = nn.Parameter(self.mean_texture.repeat(d, 1))

        self.shape_codes = self.shape_codes.to(self.device)
        self.texture_codes = self.texture_codes.to(self.device)

    def load_pretrained_model(self, saved_model_file):
        saved_data = torch.load(saved_model_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)

        # load in previously optimized codes for codenerf
        if self.hpams['arch'] == 'codenerf':
            # mean shape should only consider those optimized codes when some of those are not touched
            if 'optimized_idx' in saved_data.keys():
                optimized_idx = saved_data['optimized_idx'].numpy()
                self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'][optimized_idx > 0], dim=0).reshape(1,
                                                                                                                          -1)
                self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'][optimized_idx > 0],
                                               dim=0).reshape(1, -1)
            else:
                self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'], dim=0).reshape(1, -1)
                self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'], dim=0).reshape(1, -1)

    def make_savedir(self, save_dir):
        self.save_dir = os.path.join(f'exps_nuscenes_' + self.hpams['arch'], save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(os.path.join(self.save_dir, 'runs'))
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'runs'))
        hpampath = os.path.join(self.save_dir, 'hpam.json')
        # update the model folder
        self.hpams['model_dir'] = self.save_dir
        with open(hpampath, 'w') as f:
            json.dump(self.hpams, f, indent=2)

    def save_models(self, iter=None, epoch=None):
        if self.hpams['arch'] == 'autorf':
            save_dict = {'model_params': self.model.state_dict(),
                         'niter': self.niter,
                         'nepoch': self.nepoch,
                         }
        elif self.hpams['arch'] == 'codenerf':
            save_dict = {'model_params': self.model.state_dict(),
                         'shape_code_params': self.shape_codes.state_dict(),
                         'texture_code_params': self.texture_codes.state_dict(),
                         'niter': self.niter,
                         'nepoch': self.nepoch,
                         'instoken2idx': self.instoken2idx,
                         'optimized_idx': self.optimized_idx
                         }
        else:
            save_dict = {}
            print('ERROR: No valid network architecture is declared in config file!')

        if iter != None:
            torch.save(save_dict, os.path.join(self.save_dir, str(iter) + '.pth'))
        if epoch != None:
            torch.save(save_dict, os.path.join(self.save_dir, f'epoch_{epoch}.pth'))
        torch.save(save_dict, os.path.join(self.save_dir, 'models.pth'))

    def resume_from_epoch(self, saved_dir, epoch):
        print(f'Resume training from saved model at epoch {epoch}.')
        saved_path = os.path.join('exps_nuscenes_' + self.hpams['arch'], saved_dir, f'epoch_{epoch}.pth')
        saved_data = torch.load(saved_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)
        self.niter = saved_data['niter'] + 1
        self.nepoch = saved_data['nepoch'] + 1

        if self.hpams['arch'] == 'codenerf':
            self.shape_codes.load_state_dict(saved_data['shape_code_params'])
            self.texture_codes.load_state_dict(saved_data['texture_code_params'])
            self.instoken2idx = saved_data['instoken2idx']
            self.optimized_idx = saved_data['optimized_idx']
