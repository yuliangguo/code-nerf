import random
import numpy as np
import cv2
import torch
import argparse
import torchvision
from torchvision.transforms import Resize


def preprocess_img_keepratio(img, max_img_sz=320):
    """
        keep dim, and resize the max dim to max_img_sz if over
    """
    img = img.unsqueeze(0).permute((0, 3, 1, 2))
    _, _, im_h, im_w = img.shape
    if np.maximum(im_h, im_w) > max_img_sz:
        ratio = max_img_sz / np.maximum(im_h, im_w)
        new_h = im_h * ratio
        new_w = im_w * ratio
        img = Resize((int(new_h), int(new_w)))(img)
    return img


def preprocess_img_square(img, new_size=128, pad_white=True):
    """
        The largest dim resize to new_size, pad the other to make square
        Make the padding area white
    """
    img = img.unsqueeze(0).permute((0, 3, 1, 2))
    _, _, im_h, im_w = img.shape
    ratio = new_size / np.maximum(im_h, im_w)
    new_h = int(im_h * ratio)
    new_w = int(im_w * ratio)
    img = Resize((new_h, new_w))(img)
    if pad_white:
        new_img = torch.ones((1, 3, new_size, new_size), dtype=torch.float32)
    else:
        new_img = torch.zeros((1, 3, new_size, new_size), dtype=torch.float32)
    y_start = int(new_size/2 - new_h/2)
    x_start = int(new_size/2 - new_w/2)

    new_img[:, :, y_start: y_start + new_h, x_start: x_start + new_w] = img
    return new_img
    

def get_rays_srn(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    # TODO: such dir seems to be based on wield camera pose c2w and camera frame definition
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs


def get_rays(K, c2w, roi):
    """
        K: intrinsic matrix
        c2w: camera pose in object (world) coordinate frame
        roi: [min_x, min_y, max_x, max_y]

        ATTENTION:
        the number of output rays depends on roi inputs
        nuscenes uses a different camera coordinate frame compared to shapenet srn
    """
    dx = K[0, 2]
    dy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    i, j = torch.meshgrid(torch.linspace(roi[0], roi[2]-1, roi[2]-roi[0]),
                          torch.linspace(roi[1], roi[3]-1, roi[3]-roi[1]))
    i = i.t()
    j = j.t()
    # some signs are opposite to get_rays for shapenet srn dataset
    dirs = torch.stack([(i - dx) / fx, (j - dy) / fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs


def sample_from_rays(ro, vd, near, far, N_samples, z_fixed=False):
    # Given ray centre (camera location), we sample z_vals
    # TODO: this type of sampling might be limited to the camera view facing the object center.
    #  The samples from rays away from the image center can be very sparse
    # we do not use ray_o here - just number of rays
    if z_fixed:
        z_vals = torch.linspace(near, far, N_samples).type_as(ro)
    else:
        dist = (far - near) / (2*N_samples)
        z_vals = torch.linspace(near+dist, far-dist, N_samples).type_as(ro)
        z_vals += (torch.rand(N_samples) * (far - near) / (2*N_samples)).type_as(ro)
    xyz = ro.unsqueeze(-2) + vd.unsqueeze(-2) * z_vals.unsqueeze(-1)
    vd = vd.unsqueeze(-2).repeat(1,N_samples,1)
    return xyz, vd, z_vals


def volume_rendering(sigmas, rgbs, z_vals):
    deltas = z_vals[1:] - z_vals[:-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:1]) * 1e10])
    alphas = 1 - torch.exp(-sigmas.squeeze(-1) * deltas)
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -1)

    return rgb_final, depth_final


def volume_rendering2(sigmas, rgbs, z_vals):
    """
        return accumulated transparency in addition
    """
    deltas = z_vals[1:] - z_vals[:-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:1]) * 1e10])
    alphas = 1 - torch.exp(-sigmas.squeeze(-1) * deltas)
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -1)
    return rgb_final, depth_final, accum_trans[:, -1]


def volume_rendering_batch(sigmas, rgbs, z_vals):
    """
        consider the first dimension for batch
    """
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:, :1]) * 1e10], -1)
    alphas = 1 - torch.exp(-sigmas.squeeze(-1) * deltas.unsqueeze(1))
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals.unsqueeze(1), -1)
    return rgb_final, depth_final, accum_trans[:, :, -1]


def prepare_pixel_samples(img, mask_occ, cam_pose, obj_diag, K, roi, n_rays, n_samples, shapenet_obj_cood, sym_aug):
    """
        Prepare pixel-sampled data from input image
    """
    # near and far sample range need to be adaptively calculated
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2

    rays_o, viewdir = get_rays(K, cam_pose, roi)

    # For different sized roi, extract a random subset of pixels with fixed batch size
    n_rays = np.minimum(rays_o.shape[0], n_rays)
    random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
    rays_o = rays_o[random_ray_ids]
    viewdir = viewdir[random_ray_ids]

    # extract samples
    rgb_tgt = img.reshape(-1, 3)[random_ray_ids]
    occ_pixels = mask_occ.reshape(-1, 1)[random_ray_ids]
    mask_rgb = torch.clone(mask_occ)
    mask_rgb[mask_rgb < 0] = 0

    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
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

    return xyz, viewdir, z_vals, rgb_tgt, occ_pixels


def render_rays(model, device, img, mask_occ, cam_pose, obj_diag, K, roi, n_rays, n_samples, shapecode, texturecode, shapenet_obj_cood, sym_aug):
    """
        Assume only one input image, sample pixels from the roi area, and render rgb and depth values of the sampled pixels.
        Return both rendered values and tgt values for the sampled pixels, as well additional output for training purpose
    """
    
    # near and far sample range need to be adaptively calculated
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2

    rays_o, viewdir = get_rays(K, cam_pose, roi)

    # For different sized roi, extract a random subset of pixels with fixed batch size
    n_rays = np.minimum(rays_o.shape[0], n_rays)
    random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
    rays_o = rays_o[random_ray_ids]
    viewdir = viewdir[random_ray_ids]

    # extract samples
    rgb_tgt = img.reshape(-1, 3)[random_ray_ids].to(device)
    occ_pixels = mask_occ.reshape(-1, 1)[random_ray_ids].to(device)
    mask_rgb = torch.clone(mask_occ)
    mask_rgb[mask_rgb < 0] = 0

    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
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

    sigmas, rgbs = model(xyz.to(device),
                         viewdir.to(device),
                         shapecode, texturecode)
    rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(device))
    return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels


def render_full_img(model, device, cam_pose, obj_sz, K, roi, n_samples, shapecode, texturecode, shapenet_obj_cood, debug_occ=False):
    """
        Assume only one input image, render rgb and depth values of the all the image pixels within the roi area.
        Only the rendered image is returned for visualization purpose.
    """
    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

    # near and far sample range need to be adaptively calculated
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2

    rays_o, viewdir = get_rays(K, cam_pose, roi)

    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
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
        sigmas, rgbs = model(xyz[i:i + sample_step].to(device),
                                  viewdir[i:i + sample_step].to(device),
                                  shapecode, texturecode)
        if debug_occ:
            rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(device))
            generated_acc_trans.append(acc_trans_rays)
        else:
            rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(device))
        generated_img.append(rgb_rays)
    generated_img = torch.cat(generated_img).reshape(roi[3] - roi[1], roi[2] - roi[0], 3)

    if debug_occ:
        generated_acc_trans = torch.cat(generated_acc_trans).reshape(roi[3]-roi[1], roi[2]-roi[0])
        cv2.imshow('est_occ', ((torch.ones_like(generated_acc_trans) - generated_acc_trans).cpu().numpy() * 255).astype(np.uint8))
        # cv2.imshow('mask_occ', ((gt_masks_occ[0].cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8))
        cv2.waitKey()

    return generated_img


def render_virtual_imgs(model, device, obj_sz, K, n_samples, shapecode, texturecode, shapenet_obj_cood, radius=40., tilt=np.pi/6, pan_num=8, img_sz=128):
    """
        Given NeRF model and conditioned shapecode and texturecode, render a set of virtual images from different views
    """
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
        generated_img = render_full_img(model, device, cam_pose, obj_sz, K, roi, n_samples, shapecode, texturecode, shapenet_obj_cood)
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


def calc_cam_pose_err(est_cam_poses, tgt_cam_poses):
    est_R = est_cam_poses[:, :3, :3]
    est_T = est_cam_poses[:, :3, 3]
    tgt_R = tgt_cam_poses[:, :3, :3]
    tgt_T = tgt_cam_poses[:, :3, 3]

    err_R = rot_dist(est_R, tgt_R)
    err_T = torch.sqrt(torch.sum((est_T - tgt_T) ** 2, dim=-1))
    return err_R, err_T


def calc_obj_pose_err(est_cam_poses, tgt_cam_poses):
    est_R = est_cam_poses[:, :3, :3].transpose(-1, -2)
    est_T = -torch.matmul(est_R, est_cam_poses[:, :3, 3:]).squeeze(-1)
    tgt_R = tgt_cam_poses[:, :3, :3].transpose(-1, -2)
    tgt_T = -torch.matmul(tgt_R, tgt_cam_poses[:, :3, 3:]).squeeze(-1)

    err_R = rot_dist(est_R, tgt_R)
    err_T = torch.sqrt(torch.sum((est_T - tgt_T) ** 2, dim=-1))
    return err_R, err_T


def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    #print(img.shape)
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def rot_dist(R1, R2):
    """
    R1: B x 3 x 3
    R2: B x 3 x 3
    return B X 1
    """
    R_diff = torch.matmul(R1, torch.transpose(R2, -1, -2))
    trace = torch.tensor([torch.trace(R_diff_single) for R_diff_single in R_diff])
    return torch.acos((trace-1) / 2)


def generate_obj_sz_reg_samples(obj_sz, obj_diag, shapenet_obj_cood=True, tau=0.05, samples_per_plane=100):
    """
        Generate samples around limit planes
    """
    norm_limits = obj_sz / obj_diag
    if shapenet_obj_cood:
        norm_limits = norm_limits[[1, 0, 2]]  # the limit does not care the sign
    x_lim, y_lim, z_lim = norm_limits
    out_samples = {}
    X = np.random.uniform(-x_lim, x_lim, samples_per_plane)
    Y = np.random.uniform(-y_lim, y_lim, samples_per_plane)
    Z = np.random.uniform(-z_lim, z_lim, samples_per_plane)

    out_samples['X_planes_out'] = np.concatenate([np.asarray([np.ones(samples_per_plane) * (-x_lim - tau), Y, Z]).transpose(),
                                                np.asarray([np.ones(samples_per_plane) * (x_lim + tau), Y, Z]).transpose()],
                                               axis=0).astype(np.float32)
    out_samples['X_planes_in'] = np.concatenate([np.asarray([np.ones(samples_per_plane) * (-x_lim + tau), Y, Z]).transpose(),
                                                np.asarray([np.ones(samples_per_plane) * (x_lim - tau), Y, Z]).transpose()],
                                               axis=0).astype(np.float32)

    out_samples['Y_planes_out'] = np.concatenate([np.asarray([X, np.ones(samples_per_plane) * (-y_lim - tau), Z]).transpose(),
                                                np.asarray([X, np.ones(samples_per_plane) * (y_lim + tau), Z]).transpose()],
                                               axis=0).astype(np.float32)
    out_samples['Y_planes_in'] = np.concatenate([np.asarray([X, np.ones(samples_per_plane) * (-y_lim + tau), Z]).transpose(),
                                                np.asarray([X, np.ones(samples_per_plane) * (y_lim - tau), Z]).transpose()],
                                               axis=0).astype(np.float32)

    out_samples['Z_planes_out'] = np.concatenate([np.asarray([X, Y, np.ones(samples_per_plane) * (-z_lim - tau)]).transpose(),
                                                np.asarray([X, Y, np.ones(samples_per_plane) * (z_lim + tau)]).transpose()],
                                               axis=0).astype(np.float32)
    out_samples['Z_planes_in'] = np.concatenate([np.asarray([X, Y, np.ones(samples_per_plane) * (-z_lim + tau)]).transpose(),
                                                np.asarray([X, Y, np.ones(samples_per_plane) * (z_lim - tau)]).transpose()],
                                               axis=0).astype(np.float32)
    return out_samples


def align_imgs_width(imgs, W, max_view=4):
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