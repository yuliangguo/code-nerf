
import imageio
import numpy as np
import torch
# import json
# from torchvision import transforms
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from cityscapesscripts.helpers.labels import labels, name2label


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


def get_tgt_ins(pan, cat_id, box, divisor=1000):
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

    tgt_ins_id = ins_ids[max_id]
    tgt_pixels_in_box = cnts[max_id]
    area_ratio = float(tgt_pixels_in_box) / box_area
    return tgt_ins_id, tgt_pixels_in_box, area_ratio, box_iou

#
# def get_tgt_ins(pan, cat_id, box, divisor=1000):
#     # return the instance label and pixel counts in the box
#     min_x, min_y, max_x, max_y = box
#     box_pan = pan[int(min_y):int(max_y), int(min_x):int(max_x)]
#     ins_ids, cnts = np.unique(box_pan, return_counts=True)
#     tgt_ins_id = np.where((ins_ids // divisor) == cat_id)[0]
#     if len(tgt_ins_id) == 0:
#         return None, 0, 0, 0
#
#     ins_ids = ins_ids[tgt_ins_id]
#     cnts = cnts[tgt_ins_id]
#
#     max_id = np.argmax(cnts)
#     tgt_ins_id = ins_ids[max_id]
#     tgt_pixels_in_box = cnts[max_id]
#     box_area = (max_x - min_x) * (max_y - min_y)
#     area_ratio = float(tgt_pixels_in_box) / box_area
#
#     # calculate box iou with the full instance mask (aim to remove occluded case)
#     ins_y, ins_x = np.where(pan == tgt_ins_id)
#     min_x2 = np.min(ins_x)
#     max_x2 = np.max(ins_x)
#     min_y2 = np.min(ins_y)
#     max_y2 = np.max(ins_y)
#
#     x_left = max(min_x, min_x2)
#     y_top = max(min_y, min_y2)
#     x_right = min(max_x, max_x2)
#     y_bottom = min(max_y, max_y2)
#
#     if x_right < x_left or y_bottom < y_top:
#         return tgt_ins_id, tgt_pixels_in_box, area_ratio, 0.0
#
#     intersection = (x_right - x_left) * (y_bottom - y_top)
#     union = box_area + (max_x2 - min_x2) * (max_y2 - min_y2) - intersection
#     box_iou = intersection/union
#     return tgt_ins_id, tgt_pixels_in_box, area_ratio, box_iou


class NuScenesData:
    def __init__(self, nusc_cat='vehicle.car',
                 cs_cat='car',
                 nusc_data_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini',
                 nusc_pan_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/panoptic_pred',
                 nvsc_version='v1.0-mini',
                 num_cams_per_sample=1,
                 divisor=1000,
                 box_iou_th=0.5,
                 mask_pixels=2500,
                 img_h=900,
                 img_w=1600,
                 debug=False):
        """
            Provide camera input and label per annotation per instance for the target category
            Object 'Instance' here respects the definition from NuScene dataset.
            Each instance of object does not go beyond a single scene.
            Each instance contains multiple annotations from different timestamps
            Each annotation could be projected to multiple camera inputs at the same timestamp.
        """
        self.nusc_cat = nusc_cat
        self.cs_cat = cs_cat
        self.divisor = divisor
        self.box_iou_th = box_iou_th
        self.mask_pixels = mask_pixels
        self.img_h = img_h
        self.img_w = img_w
        self.debug = debug

        self.nusc_data_dir = nusc_data_dir
        self.nusc_pan_dir = nusc_pan_dir
        self.nusc = NuScenes(version=nvsc_version, dataroot=nusc_data_dir, verbose=True)
        # self.nusc.list_scenes()
        instance_all = self.nusc.instance
        # self.tgt_instance_list = []
        self.instokens = []
        self.anntokens = []  # multiple anntokens can have the same instoken
        # retrieve all the target instance
        for instance in instance_all:
            if self.nusc.get('category', instance['category_token'])['name'] == nusc_cat:
                # self.tgt_instance_list.append(instance)
                anntokens = self.nusc.field2token('sample_annotation', 'instance_token', instance['token'])
                for anntoken in anntokens:
                    self.instokens.append(instance['token'])
                    self.anntokens.append(anntoken)
        self.lenids = len(self.anntokens)
        print(f'{self.lenids} annotations in {self.nusc_cat} category are included in dataloader.')
        self.num_cams_per_sample = num_cams_per_sample

    def __len__(self):
        return self.lenids
    
    def __getitem__(self, idx):
        instoken = self.instokens[idx]
        anntoken = self.anntokens[idx]
        if self.debug:
            print(f'instance: {instoken}, anntoken: {anntoken}')

        # extract fixed number of qualified samples per instance
        imgs = []
        masks = []
        camera_poses = []
        camera_intrinsics = []
        rois = []  # used to sample rays
        valid_flags = []

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

                    # visualize pred instance mask
                    pan_file = os.path.join(self.nusc_pan_dir, cam, 'panoptic', os.path.basename(data_path)[:-4] + '.png')
                    pan_img = np.asarray(Image.open(pan_file))
                    pan_label = img2pan(pan_img)

                    # box here is in sensor coordinate system
                    box = boxes[0]
                    # compute the camera pose in object frame, make sure dataset and model definitions consistent
                    obj_center = box.center
                    obj_orientation = box.orientation.rotation_matrix

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
                    tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou = get_tgt_ins(pan_label,
                                                                               name2label[self.cs_cat][2],
                                                                               box_2d,
                                                                               self.divisor)
                    if self.debug:
                        print(
                            f'        tgt instance id: {tgt_ins_id}, '
                            f'num of pixels: {tgt_ins_cnt}, '
                            f'area ratio: {area_ratio}, '
                            f'box_iou: {box_iou}')
                    if tgt_ins_id is not None and tgt_ins_cnt > self.mask_pixels and box_iou > self.box_iou_th:
                        imgs.append(img)
                        masks.append((pan_label == tgt_ins_id).astype(np.int32))
                        rois.append(box_2d)
                        camera_intrinsics.append(camera_intrinsic)
                        camera_poses.append(cam_pose)
                        valid_flags.append(1)

                    if self.debug:
                        camtoken = sample_record['data'][cam]
                        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
                        axes[0].imshow(img)
                        axes[0].set_title(self.nusc.get('sample_data', camtoken)['channel'])
                        axes[0].axis('off')
                        axes[0].set_aspect('equal')
                        c = np.array(self.nusc.colormap[box.name]) / 255.0
                        box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c))

                        ins_vis = pan2ins_vis(pan_label, name2label[self.cs_cat][2], self.divisor)
                        axes[1].imshow(ins_vis)
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
                masks.append(np.zeros((self.img_h, self.img_w, 3)))
                rois.append(np.array([-1, -1, -1, -1]))
                camera_intrinsics.append(np.zeros((3, 3)).astype(np.float32))
                camera_poses.append(np.zeros((3, 4)).astype(np.float32))
                valid_flags.append(0)

        return torch.from_numpy(np.asarray(imgs)), \
               torch.from_numpy(np.asarray(masks)), \
               torch.from_numpy(np.asarray(rois).astype(np.int32)), \
               torch.from_numpy(np.asarray(camera_intrinsics).astype(np.float32)), \
               torch.from_numpy(np.asarray(camera_poses).astype(np.float32)), \
               np.asarray(valid_flags), instoken, anntoken


if __name__ == '__main__':
    # TODO: check if instance mask can be found from nuimages
    from torch.utils.data import DataLoader

    nusc_dataset = NuScenesData(
        nusc_cat='vehicle.car',
        cs_cat='car',
        nusc_data_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini',
        nusc_pan_dir='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/panoptic_pred',
        nvsc_version='v1.0-mini',
        num_cams_per_sample=1,
        divisor=1000,
        box_iou_th=0.5,
        mask_pixels=2500,
        img_h=900,
        img_w=1600,
        debug=False)
    dataloader = DataLoader(nusc_dataset, batch_size=1, num_workers=0, shuffle=True)

    # Analysis of valid portion of data
    valid_ann_total = 0
    valid_ins_dic = {}
    for ii, d in enumerate(dataloader):
        imgs, masks, rois, camera_intrinsics, camera_poses, valid_flags, instoken, anntoken = d
        num_valid_cam = np.sum(valid_flags.numpy())
        valid_ann_total += int(num_valid_cam > 0)
        if instoken[0] not in valid_ins_dic.keys():
            valid_ins_dic[instoken[0]] = 0
        if num_valid_cam > 0:
            valid_ins_dic[instoken[0]] = 1
        print(f'Finish {ii} / {len(dataloader)}, valid samples: {num_valid_cam}')
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
