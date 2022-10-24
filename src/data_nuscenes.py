import random
import imageio
import numpy as np
import torch
import json
# from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from cityscapesscripts.helpers.labels import labels, name2label, trainId2label

import data_splits_nusc

# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L14
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


# label encoding based on cityscape format
def pan2img(pan):
    # unique image encoding of a large iD int into RGB channels
    img = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = pan % 256
    img[:, :, 1] = (pan // 256) % 256
    img[:, :, 2] = (pan // 256 // 256) % 256
    return img


# label encoding based on cityscape format
def img2pan(img):
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3
    pan = img[:, :, 0] + 256 * img[:, :, 1] + 256 * 256 * img[:, :, 2]
    return pan


def pan2ins_vis(pan, cat_id, divisor):
    img = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.float32)
    pixels = (pan // divisor) == cat_id
    if np.sum(pixels) == 0:
        return img
    ins_ids = np.unique(pan[pixels])
    for ins_id in ins_ids:
        yy, xx = np.where(pan == ins_id)
        img[yy, xx, :] = _COLORS[ins_id % divisor]
    return img


def ins2vis(masks):
    masks = np.asarray(masks)
    img = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.float32)

    for ins_id, mask in enumerate(masks):
        yy, xx = np.where(mask > 0)
        img[yy, xx, :] = _COLORS[ins_id % _COLORS.shape[0]]
    return img


def get_mask_occ_cityscape(pan, tgt_pan_id, divisor):
    """
        Prepare occupancy mask:
            target object: 1
            background: -1 (not likely to occlude the object)
            occluded the instance: 0 (seems only able to reason the occlusion by foreground)
    """
    mask_occ = np.zeros_like(pan).astype(np.int32)
    pan_ids = np.unique(pan)

    # assign background
    for pan_id in pan_ids:
        cat_id = pan_id // divisor
        if trainId2label[cat_id].category in ['sky', 'nature', 'construction', 'flat', 'void']:
            mask_occ[pan == pan_id] = -1
        elif pan_id == tgt_pan_id:
            mask_occ[pan == pan_id] = 1
    return mask_occ


def get_mask_occ_from_ins(masks, tgt_ins_id):
    tgt_mask = masks[tgt_ins_id]
    mask_occ = np.zeros_like(tgt_mask).astype(np.int32)
    mask_union = np.sum(np.asarray(masks), axis=0)

    mask_occ[mask_union == 0] = -1
    mask_occ[tgt_mask > 0] = 1
    return mask_occ


def get_tgt_ins_from_pan(pan, cat_id, box, divisor=1000):
    # return the instance label and pixel counts in the box
    min_x, min_y, max_x, max_y = box
    box_pan = pan[int(min_y):int(max_y), int(min_x):int(max_x)]
    ins_ids, cnts = np.unique(box_pan, return_counts=True)
    tgt_ins_id = np.where((ins_ids // divisor) == cat_id)[0]
    if len(tgt_ins_id) == 0:
        return None, 0, 0, 0

    ins_ids = ins_ids[tgt_ins_id]
    cnts = cnts[tgt_ins_id]
    box_area = (max_x - min_x) * (max_y - min_y)

    max_id = 0
    box_iou = 0.0
    area_ratio = 0.0

    for ii, ins_id in enumerate(ins_ids):
        # calculate box iou with the full instance mask (aim to remove occluded case)
        ins_y, ins_x = np.where(pan == ins_id)
        min_x2 = np.min(ins_x)
        max_x2 = np.max(ins_x)
        min_y2 = np.min(ins_y)
        max_y2 = np.max(ins_y)

        x_left = max(min_x, min_x2)
        y_top = max(min_y, min_y2)
        x_right = min(max_x, max_x2)
        y_bottom = min(max_y, max_y2)

        if x_right < x_left or y_bottom < y_top:
            box_iou_i = 0.0
        else:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            union = box_area + (max_x2 - min_x2) * (max_y2 - min_y2) - intersection
            box_iou_i = intersection/union

        if box_iou_i > box_iou:
            max_id = ii
            box_iou = box_iou_i
            area_ratio = float(len(ins_y)) / (max_x2 - min_x2) / (max_y2 - min_y2)

    tgt_ins_id = ins_ids[max_id]
    tgt_pixels_in_box = cnts[max_id]
    # area_ratio = float(tgt_pixels_in_box) / box_area
    return tgt_ins_id, tgt_pixels_in_box, area_ratio, box_iou


# TODO: iou can be calculated faster
def get_tgt_ins_from_pred(preds, masks, tgt_cat, tgt_box):
    # locate the detections matched the target category
    indices = [idx for idx, label in enumerate(preds['labels']) if tgt_cat in label]
    if len(indices) == 0:
        return 0, 0, 0., 0.

    boxes = np.asarray(preds['boxes'])[indices]

    # calculate box ious between predicted boxed and tgt box
    min_x, min_y, max_x, max_y = tgt_box
    box_area = (max_x - min_x) * (max_y - min_y)
    max_id = 0
    box_iou = 0.0
    for ii, pred_box in enumerate(boxes):
        min_x2, min_y2, max_x2, max_y2 = pred_box
        x_left = max(min_x, min_x2)
        y_top = max(min_y, min_y2)
        x_right = min(max_x, max_x2)
        y_bottom = min(max_y, max_y2)

        if x_right < x_left or y_bottom < y_top:
            box_iou_i = 0.0
        else:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            union = box_area + (max_x2 - min_x2) * (max_y2 - min_y2) - intersection
            box_iou_i = intersection / union

        if box_iou_i > box_iou:
            max_id = ii
            box_iou = box_iou_i

    tgt_ins_id = indices[max_id]
    tgt_mask = masks[tgt_ins_id]
    tgt_ins_cnt = np.sum((tgt_mask > 0).astype(np.int))
    tgt_box = boxes[max_id]
    tgt_bb_area = (tgt_box[2] - tgt_box[0]) * (tgt_box[3] - tgt_box[1])
    area_ratio = float(tgt_ins_cnt) / tgt_bb_area
    return tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou


# TODO: make an initial pre-process save all the valid cases?
class NuScenesData:
    def __init__(self, nusc_cat='vehicle.car',
                 seg_cat='car',
                 nusc_data_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini',
                 nusc_seg_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/pred_panoptic',
                 nusc_version='v1.0-mini',
                 split='train',
                 num_cams_per_sample=1,
                 divisor=1000,
                 box_iou_th=0.5,
                 max_dist=40,
                 mask_pixels=10000,
                 img_h=900,
                 img_w=1600,
                 add_pose_err=False,
                 max_rot_pert=0.2,
                 max_t_pert=0.1,
                 debug=False,
                 ):
        """
            Provide camera input and label per annotation per instance for the target category
            Object 'Instance' here respects the definition from NuScene dataset.
            Each instance of object does not go beyond a single scene.
            Each instance contains multiple annotations from different timestamps
            Each annotation could be projected to multiple camera inputs at the same timestamp.
        """
        self.nusc_cat = nusc_cat
        self.seg_cat = seg_cat
        self.divisor = divisor
        self.box_iou_th = box_iou_th
        self.max_dist = max_dist
        self.mask_pixels = mask_pixels
        self.img_h = img_h
        self.img_w = img_w
        self.debug = debug

        self.nusc_data_dir = nusc_data_dir
        self.nusc_seg_dir = nusc_seg_dir
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_data_dir, verbose=True)
        # self.nusc.list_scenes()
        instance_all = self.nusc.instance
        # self.tgt_instance_list = []
        self.instokens = []
        self.anntokens = []  # multiple anntokens can have the same instoken
        self.ins_ann_tokens = {}
        # save and load to avoid repeating preparation
        subset_file = 'jsonfiles/nusc.' + nusc_version + '.' + split + '.' + nusc_cat + '.json'
        if os.path.exists(subset_file):
            nusc_subset = json.load(open(subset_file))
            self.instokens = nusc_subset['instokens']
            self.anntokens = nusc_subset['anntokens']
            self.ins_ann_tokens = nusc_subset['ins_ann_tokens']
            print('Loaded pre-prepared subset of instance and annotation lists.')
        else:
            # retrieve all the target instance
            print('Preparing related subset of instance and annotation lists ...')
            for instance in tqdm(instance_all):
                if self.nusc.get('category', instance['category_token'])['name'] == nusc_cat:
                    instoken = instance['token']
                    # self.tgt_instance_list.append(instance)
                    anntokens = self.nusc.field2token('sample_annotation', 'instance_token', instoken)
                    for anntoken in anntokens:
                        # rule out those night samples
                        sample_ann = self.nusc.get('sample_annotation', anntoken)
                        sample_record = self.nusc.get('sample', sample_ann['sample_token'])
                        scene = self.nusc.get('scene', sample_record['scene_token'])
                        if 'mini' in nusc_version:
                            if split == 'train' and scene['name'] not in data_splits_nusc.mini_train:
                                continue
                            if split == 'val' and scene['name'] not in data_splits_nusc.mini_val:
                                continue
                        if 'trainval' in nusc_version:
                            if split == 'train' and scene['name'] not in data_splits_nusc.train:
                                continue
                            if split == 'val' and scene['name'] not in data_splits_nusc.val:
                                continue
                        if 'test' in nusc_version:
                            if split == 'test' and scene['name'] not in data_splits_nusc.test:
                                continue

                        log_file = self.nusc.get('log', scene['log_token'])['logfile']
                        log_items = log_file.split('-')
                        if int(log_items[4]) >= 18:  # Consider time after 18:00 as night
                            continue
                        self.instokens.append(instoken)
                        self.anntokens.append(anntoken)
                        if instoken not in self.ins_ann_tokens.keys():
                            self.ins_ann_tokens[instoken] = anntokens
            # save into json file for quick load next time
            nusc_subset = {}
            nusc_subset['instokens'] = self.instokens
            nusc_subset['anntokens'] = self.anntokens
            nusc_subset['ins_ann_tokens'] = self.ins_ann_tokens
            json.dump(nusc_subset, open(subset_file, 'w'), indent=4)

        self.lenids = len(self.anntokens)
        print(f'{self.lenids} annotations in {self.nusc_cat} category are included in dataloader.')
        self.num_cams_per_sample = num_cams_per_sample

        # for adding error to pose
        self.add_pose_err = add_pose_err
        if self.add_pose_err:
            self.max_rot_pert = max_rot_pert
            self.max_t_pert = max_t_pert

    def __len__(self):
        return self.lenids
    
    def __getitem__(self, idx):
        sample_data = {}
        instoken = self.instokens[idx]
        anntoken = self.anntokens[idx]
        if self.debug:
            print(f'instance: {instoken}, anntoken: {anntoken}')

        # extract fixed number of qualified samples per instance
        imgs = []
        masks_occ = []
        cam_poses = []
        cam_intrinsics = []
        rois = []  # used to sample rays
        valid_flags = []
        cam_poses_w_err = []

        # TODO: apply different parsing depending on the segmentation type
        # For each annotation (one annotation per timestamp) get all the sensors
        sample_ann = self.nusc.get('sample_annotation', anntoken)
        sample_record = self.nusc.get('sample', sample_ann['sample_token'])
        if 'LIDAR_TOP' in sample_record['data'].keys():
            # Figure out which camera the object is fully visible in (this may return nothing).
            cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
            cams = np.random.permutation(cams)
            for cam in cams:
                if self.debug:
                    print(f'     cam{cam}')
                # TODO: consider BoxVisibility.ANY?
                data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam],
                                                                               box_vis_level=BoxVisibility.ALL,
                                                                               selected_anntokens=[anntoken])
                if len(boxes) == 1:
                    # Plot CAMERA view.
                    img = Image.open(data_path)
                    img = np.asarray(img)

                    # box here is in sensor coordinate system
                    box = boxes[0]
                    # compute the camera pose in object frame, make sure dataset and model definitions consistent
                    obj_center = box.center
                    obj_orientation = box.orientation.rotation_matrix

                    # ATTENTION: add Rot error in the object's coordinate, and T error
                    if self.add_pose_err:
                        # only consider yaw error and distance error
                        yaw_err = random.uniform(-self.max_rot_pert, self.max_rot_pert)
                        rot_err = np.array([[np.cos(yaw_err), -np.sin(yaw_err), 0.],
                                            [np.sin(yaw_err), np.cos(yaw_err), 0.],
                                            [0., 0., 1.]]).astype(np.float32)
                        trans_err_ratio = random.uniform(1.0-self.max_t_pert, 1.0+self.max_t_pert)
                        obj_center_w_err = obj_center * trans_err_ratio
                        obj_orientation_w_err = obj_orientation @ rot_err
                        R_c2o_w_err = obj_orientation_w_err.transpose()
                        t_c2o_w_err = -R_c2o_w_err @ np.expand_dims(obj_center_w_err, -1)
                        cam_pose_w_err = np.concatenate([R_c2o_w_err, t_c2o_w_err], axis=1)
                        # TODO: not synced with box_2d

                    # Compute camera pose in object frame = c2o transformation matrix
                    # Recall that object pose in camera frame = o2c transformation matrix
                    R_c2o = obj_orientation.transpose()
                    t_c2o = - R_c2o @ np.expand_dims(obj_center, -1)
                    cam_pose = np.concatenate([R_c2o, t_c2o], axis=1)

                    # find the valid instance given 2d box projection
                    corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
                    min_x = np.min(corners[0, :])
                    max_x = np.max(corners[0, :])
                    min_y = np.min(corners[1, :])
                    max_y = np.max(corners[1, :])
                    box_2d = [min_x, min_y, max_x, max_y]

                    if 'panoptic' in self.nusc_seg_dir:
                        pan_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + '.png')
                        pan_img = np.asarray(Image.open(pan_file))
                        pan_label = img2pan(pan_img)
                        tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou = get_tgt_ins_from_pan(pan_label,
                                                                                            name2label[self.seg_cat][2],
                                                                                            box_2d,
                                                                                            self.divisor)
                        mask_occ = get_mask_occ_cityscape(pan_label, tgt_ins_id, self.divisor)

                    elif 'instance' in self.nusc_seg_dir:
                        json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + '.json')
                        preds = json.load(open(json_file))
                        ins_masks = []
                        for box_id in range(0, len(preds['boxes'])):
                            mask_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + f'_{box_id}.png')
                            mask = np.asarray(Image.open(mask_file))
                            ins_masks.append(mask)

                        tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou = get_tgt_ins_from_pred(preds,
                                                                                             ins_masks,
                                                                                             self.seg_cat,
                                                                                             box_2d)
                        if len(ins_masks) == 0:
                            mask_occ = None
                        else:
                            mask_occ = get_mask_occ_from_ins(ins_masks, tgt_ins_id)

                    if tgt_ins_id is not None and tgt_ins_cnt > self.mask_pixels and box_iou > self.box_iou_th and area_ratio > self.box_iou_th and np.linalg.norm(obj_center) < self.max_dist:
                        imgs.append(img)
                        masks_occ.append(mask_occ.astype(np.int32))
                        # masks.append((pan_label == tgt_ins_id).astype(np.int32))
                        rois.append(box_2d)
                        cam_intrinsics.append(camera_intrinsic)
                        cam_poses.append(cam_pose)
                        valid_flags.append(1)

                        if self.add_pose_err:
                            cam_poses_w_err.append(cam_pose_w_err)

                        if self.debug:
                            print(
                                f'        tgt instance id: {tgt_ins_id}, '
                                f'num of pixels: {tgt_ins_cnt}, '
                                f'area ratio: {area_ratio}, '
                                f'box_iou: {box_iou}')

                            camtoken = sample_record['data'][cam]
                            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
                            axes[0].imshow(img)
                            axes[0].set_title(self.nusc.get('sample_data', camtoken)['channel'])
                            axes[0].axis('off')
                            axes[0].set_aspect('equal')
                            c = np.array(self.nusc.colormap[box.name]) / 255.0
                            box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c))

                            if 'panoptic' in self.nusc_seg_dir:
                                seg_vis = pan2ins_vis(pan_label, name2label[self.seg_cat][2], self.divisor)
                            elif 'instance' in self.nusc_seg_dir:
                                seg_vis = ins2vis(ins_masks)
                            axes[1].imshow(seg_vis)
                            axes[1].set_title('pred instance')
                            axes[1].axis('off')
                            axes[1].set_aspect('equal')
                            # c = np.array(nusc.colormap[box.name]) / 255.0
                            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                                     linewidth=2, edgecolor='y', facecolor='none')
                            axes[1].add_patch(rect)
                            plt.show()

                if len(imgs) == self.num_cams_per_sample:
                    break

        # fill the insufficient data slots
        if 'LIDAR_TOP' not in sample_record['data'].keys() or len(imgs) < self.num_cams_per_sample:
            for ii in range(len(imgs), self.num_cams_per_sample):
                imgs.append(np.zeros((self.img_h, self.img_w, 3)))
                masks_occ.append(np.zeros((self.img_h, self.img_w)))
                rois.append(np.array([-1, -1, -1, -1]))
                cam_intrinsics.append(np.zeros((3, 3)).astype(np.float32))
                cam_poses.append(np.zeros((3, 4)).astype(np.float32))
                valid_flags.append(0)

                if self.add_pose_err:
                    cam_poses_w_err.append(np.zeros((3, 4)).astype(np.float32))

        sample_data['imgs'] = torch.from_numpy(np.asarray(imgs).astype(np.float32)/255.)
        sample_data['masks_occ'] = torch.from_numpy(np.asarray(masks_occ).astype(np.float32))
        sample_data['rois'] = torch.from_numpy(np.asarray(rois).astype(np.int32))
        sample_data['cam_intrinsics'] = torch.from_numpy(np.asarray(cam_intrinsics).astype(np.float32))
        sample_data['cam_poses'] = torch.from_numpy(np.asarray(cam_poses).astype(np.float32))
        sample_data['valid_flags'] = np.asarray(valid_flags)
        sample_data['instoken'] = instoken
        sample_data['anntoken'] = anntoken

        if self.add_pose_err:
            sample_data['cam_poses_w_err'] = torch.from_numpy(np.asarray(cam_poses_w_err).astype(np.float32))

        return sample_data

        # imgs = torch.from_numpy(np.asarray(imgs).astype(np.float32)/255.)
        # masks_occ = torch.from_numpy(np.asarray(masks_occ).astype(np.float32))
        # rois = torch.from_numpy(np.asarray(rois).astype(np.int32))
        # cam_intrinsics = torch.from_numpy(np.asarray(cam_intrinsics).astype(np.float32))
        # cam_poses = torch.from_numpy(np.asarray(cam_poses).astype(np.float32))
        # valid_flags = np.asarray(valid_flags)
        # instoken = instoken
        # anntoken = anntoken
        #
        # return imgs, masks_occ, rois, cam_intrinsics, cam_poses, valid_flags, instoken, anntoken

    def get_ins_samples(self, instoken):
        anntokens = self.ins_ann_tokens[instoken]

        # extract fixed number of qualified samples per instance
        imgs = []
        masks_occ = []
        cam_poses = []
        cam_poses_w_err = []
        cam_intrinsics = []
        rois = []  # used to sample rays
        out_anntokens = []

        for anntoken in anntokens:
            if self.debug:
                print(f'instance: {instoken}, anntoken: {anntoken}')

            # For each annotation (one annotation per timestamp) get all the sensors
            sample_ann = self.nusc.get('sample_annotation', anntoken)
            sample_record = self.nusc.get('sample', sample_ann['sample_token'])
            if 'LIDAR_TOP' in sample_record['data'].keys():
                # Figure out which camera the object is fully visible in (this may return nothing).
                cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
                cams = np.random.permutation(cams)
                for cam in cams:
                    if self.debug:
                        print(f'     cam {cam}')
                    # TODO: consider BoxVisibility.ANY?
                    data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam],
                                                                                   box_vis_level=BoxVisibility.ALL,
                                                                                   selected_anntokens=[anntoken])
                    if len(boxes) == 1:
                        # Plot CAMERA view.
                        img = Image.open(data_path)
                        img = np.asarray(img)

                        # box here is in sensor coordinate system
                        box = boxes[0]
                        # compute the camera pose in object frame, make sure dataset and model definitions consistent
                        obj_center = box.center
                        obj_orientation = box.orientation.rotation_matrix

                        # ATTENTION: add Rot error in the object's coordinate, and T error
                        if self.add_pose_err:
                            # only consider yaw error and distance error
                            yaw_err = random.uniform(-self.max_rot_pert, self.max_rot_pert)
                            rot_err = np.array([[np.cos(yaw_err), -np.sin(yaw_err), 0.],
                                                [np.sin(yaw_err), np.cos(yaw_err), 0.],
                                                [0., 0., 1.]]).astype(np.float32)
                            trans_err_ratio = random.uniform(1.0 - self.max_t_pert, 1.0 + self.max_t_pert)
                            obj_center_w_err = obj_center * trans_err_ratio
                            obj_orientation_w_err = obj_orientation @ rot_err
                            R_c2o_w_err = obj_orientation_w_err.transpose()
                            # ATTENTION: t_c2o_w_err is proportional to t_c2o because added error to R
                            t_c2o_w_err = -R_c2o_w_err @ np.expand_dims(obj_center_w_err, -1)
                            cam_pose_w_err = np.concatenate([R_c2o_w_err, t_c2o_w_err], axis=1)
                            # TODO: not synced with box_2d

                        # Compute camera pose in object frame = c2o transformation matrix
                        # Recall that object pose in camera frame = o2c transformation matrix
                        R_c2o = obj_orientation.transpose()
                        t_c2o = - R_c2o @ np.expand_dims(obj_center, -1)
                        cam_pose = np.concatenate([R_c2o, t_c2o], axis=1)
                        # find the valid instance given 2d box projection
                        corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
                        min_x = np.min(corners[0, :])
                        max_x = np.max(corners[0, :])
                        min_y = np.min(corners[1, :])
                        max_y = np.max(corners[1, :])
                        box_2d = [min_x, min_y, max_x, max_y]

                        if 'panoptic' in self.nusc_seg_dir:
                            pan_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + '.png')
                            pan_img = np.asarray(Image.open(pan_file))
                            pan_label = img2pan(pan_img)
                            tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou = get_tgt_ins_from_pan(pan_label,
                                                                                                name2label[
                                                                                                    self.seg_cat][2],
                                                                                                box_2d,
                                                                                                self.divisor)
                            mask_occ = get_mask_occ_cityscape(pan_label, tgt_ins_id, self.divisor)

                        elif 'instance' in self.nusc_seg_dir:
                            json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + '.json')
                            preds = json.load(open(json_file))
                            ins_masks = []
                            for box_id in range(0, len(preds['boxes'])):
                                mask_file = os.path.join(self.nusc_seg_dir, cam,
                                                         os.path.basename(data_path)[:-4] + f'_{box_id}.png')
                                mask = np.asarray(Image.open(mask_file))
                                ins_masks.append(mask)

                            tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou = get_tgt_ins_from_pred(preds,
                                                                                                 ins_masks,
                                                                                                 self.seg_cat,
                                                                                                 box_2d)
                            if len(ins_masks) == 0:
                                mask_occ = None
                            else:
                                mask_occ = get_mask_occ_from_ins(ins_masks, tgt_ins_id)

                        if tgt_ins_id is not None and tgt_ins_cnt > self.mask_pixels and box_iou > self.box_iou_th and area_ratio > self.box_iou_th and np.linalg.norm(obj_center) < self.max_dist:
                            imgs.append(img)
                            masks_occ.append(mask_occ.astype(np.int32))
                            # masks.append((pan_label == tgt_ins_id).astype(np.int32))
                            rois.append(box_2d)
                            cam_intrinsics.append(camera_intrinsic)
                            cam_poses.append(cam_pose)
                            out_anntokens.append(anntoken)

                            if self.add_pose_err:
                                cam_poses_w_err.append(cam_pose_w_err)

                            if self.debug:
                                print(
                                    f'        tgt instance id: {tgt_ins_id}, '
                                    f'num of pixels: {tgt_ins_cnt}, '
                                    f'area ratio: {area_ratio}, '
                                    f'box_iou: {box_iou}')

                                camtoken = sample_record['data'][cam]
                                fig, axes = plt.subplots(1, 2, figsize=(18, 9))
                                axes[0].imshow(img)
                                axes[0].set_title(self.nusc.get('sample_data', camtoken)['channel'])
                                axes[0].axis('off')
                                axes[0].set_aspect('equal')
                                c = np.array(self.nusc.colormap[box.name]) / 255.0
                                box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c))

                                if 'panoptic' in self.nusc_seg_dir:
                                    seg_vis = pan2ins_vis(pan_label, name2label[self.seg_cat][2], self.divisor)
                                elif 'instance' in self.nusc_seg_dir:
                                    seg_vis = ins2vis(ins_masks)
                                axes[1].imshow(seg_vis)
                                axes[1].set_title('pred instance')
                                axes[1].axis('off')
                                axes[1].set_aspect('equal')
                                # c = np.array(nusc.colormap[box.name]) / 255.0
                                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                                         linewidth=2, edgecolor='y', facecolor='none')
                                axes[1].add_patch(rect)
                                plt.show()

                    if len(imgs) == 1:
                        break

        if len(imgs) == 0:
            if self.add_pose_err:
                return None, None, None, None, None, None, None
            return None, None, None, None, None, None

        if self.add_pose_err:
            return torch.from_numpy(np.asarray(imgs).astype(np.float32) / 255.), \
                   torch.from_numpy(np.asarray(masks_occ).astype(np.float32)), \
                   torch.from_numpy(np.asarray(rois).astype(np.int32)), \
                   torch.from_numpy(np.asarray(cam_intrinsics).astype(np.float32)), \
                   torch.from_numpy(np.asarray(cam_poses).astype(np.float32)), \
                   torch.from_numpy(np.asarray(cam_poses_w_err).astype(np.float32)), \
                   np.asarray(out_anntokens)

        return torch.from_numpy(np.asarray(imgs).astype(np.float32) / 255.), \
               torch.from_numpy(np.asarray(masks_occ).astype(np.float32)), \
               torch.from_numpy(np.asarray(rois).astype(np.int32)), \
               torch.from_numpy(np.asarray(cam_intrinsics).astype(np.float32)), \
               torch.from_numpy(np.asarray(cam_poses).astype(np.float32)), \
               np.asarray(out_anntokens)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from nuimages import NuImages

    nuim = NuImages(dataroot='/mnt/LinuxDataFast/Datasets/nuimages', version='v1.0-train', verbose=True, lazy=True)

    nusc_dataset = NuScenesData(
        nusc_cat='vehicle.car',
        seg_cat='car',
        nusc_data_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini',
        nusc_seg_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/pred_instance',
        nusc_version='v1.0-mini',
        num_cams_per_sample=1,
        divisor=1000,
        box_iou_th=0.6,
        mask_pixels=2500,
        img_h=900,
        img_w=1600,
        debug=True)
    dataloader = DataLoader(nusc_dataset, batch_size=1, num_workers=0, shuffle=True)

    # Check the overlap of nuscenes and nuimages
    for ii, anntoken in enumerate(nusc_dataset.anntokens):
        sample_ann = nusc_dataset.nusc.get('sample_annotation', anntoken)
        try:
            sample_idx_check = nuim.getind('sample', sample_ann['sample_token'])
            print('find an existence in nuimages')
        except:
            print('no match in nuimages')

    # Analysis of valid portion of data
    valid_ann_total = 0
    valid_ins_dic = {}
    # for ii, d in enumerate(dataloader):
    # imgs, masks, rois, cam_intrinsics, cam_poses, valid_flags, instoken, anntoken = d
    obj_idx = 0
    for batch_idx, batch_data in enumerate(dataloader):
        masks_occ = batch_data['masks_occ'][obj_idx]
        rois = batch_data['rois'][obj_idx]
        cam_intrinsics = batch_data['cam_intrinsics'][obj_idx]
        cam_poses = batch_data['cam_poses'][obj_idx]
        valid_flags = batch_data['valid_flags'][obj_idx]
        instoken = batch_data['instoken'][obj_idx]
        anntoken = batch_data['anntoken'][obj_idx]

        num_valid_cam = np.sum(valid_flags.numpy())
        valid_ann_total += int(num_valid_cam > 0)
        if instoken[obj_idx] not in valid_ins_dic.keys():
            valid_ins_dic[instoken[obj_idx]] = 0
        if num_valid_cam > 0:
            valid_ins_dic[instoken[obj_idx]] = 1
        print(f'Finish {batch_idx} / {len(dataloader)}, valid samples: {num_valid_cam}')
    print(f'Number of annotations having valid camera data support: {valid_ann_total} out of {len(dataloader)} annotations')

    valid_ins = [ins for ins in valid_ins_dic.keys() if valid_ins_dic[ins] == 1]
    print(f'Number of instance with having valid camera data support: {len(valid_ins)} out of {len(valid_ins_dic.keys())} instances')

    # Another loop to check the optimizable portion include parked object with valid support from other timestamp
    opt_ann_total = 0
    for ii, anntoken in enumerate(nusc_dataset.anntokens):
        sample_ann = nusc_dataset.nusc.get('sample_annotation', anntoken)
        instoken = nusc_dataset.instokens[ii]
        if valid_ins_dic[instoken] > 0:
            for att_token in sample_ann['attribute_tokens']:
                attribute = nusc_dataset.nusc.get('attribute', att_token)
                if attribute['name'] == 'vehicle.parked' or attribute['name'] == 'vehicle.stopped':
                    opt_ann_total += 1
                    break
    print(f'Number of optimizable annotations having indirect camera data support: {opt_ann_total} out of {len(dataloader)} annotations')

    """
        Observed invalid scenarios:
            night (failure of instance prediction cross-domain)
            truncation (currently not included)
            general instance prediction failure
            too far-away
            too heavy occluded (some fully occluded case's annotation may come from the projection of another time's annotations for static object)
    """
