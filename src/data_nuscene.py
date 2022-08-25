
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


def tgt_instance(pan, cat_id, box):
    # return the instance label and pixel counts in the box
    min_x, min_y, max_x, max_y = box
    box_pan = pan[int(min_y):int(max_y), int(min_x):int(max_x)]
    ins_ids, cnts = np.unique(box_pan, return_counts=True)
    tgt_ins_id = np.where((ins_ids // divisor) == cat_id)[0]
    if len(tgt_ins_id) == 0:
        return None, 0

    ins_ids = ins_ids[tgt_ins_id]
    cnts = cnts[tgt_ins_id]
    max_id = np.argmax(cnts)
    return ins_ids[max_id], cnts[max_id]


def load_poses(pose_dir, idxs=[]):
    txtfiles = np.sort([os.path.join(pose_dir, f.name) for f in os.scandir(pose_dir)])
    posefiles = np.array(txtfiles)[idxs]
    srn_coords_trans = np.diag(np.array([1, -1, -1, 1])) # SRN dataset
    poses = []
    for posefile in posefiles:
        pose = np.loadtxt(posefile).reshape(4,4)
        poses.append(pose@srn_coords_trans)
    return torch.from_numpy(np.array(poses)).float()

def load_imgs(img_dir, idxs = []):
    allimgfiles = np.sort([os.path.join(img_dir, f.name) for f in os.scandir(img_dir)])
    imgfiles = np.array(allimgfiles)[idxs]
    imgs = []
    for imgfile in imgfiles:
        img = imageio.imread(imgfile, pilmode='RGB')
        img = img.astype(np.float32)
        img /= 255.
        imgs.append(img)
    return torch.from_numpy(np.array(imgs))

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

class NuScenesData():
    def __init__(self, cat='srn_cars', splits='cars_train',
                 data_dir='../data/ShapeNet_SRN/',
                 num_instances_per_obj=1, crop_img=True):
        """
        cat: srn_cars / srn_chairs
        split: cars_train(/test/val) or chairs_train(/test/val)
        First, we choose the id
        Then, we sample images (the number of instances matter)
        """
        self.data_dir = os.path.join(data_dir, cat, splits)
        self.ids = np.sort([f.name for f in os.scandir(self.data_dir)])
        self.lenids = len(self.ids)
        self.num_instances_per_obj = num_instances_per_obj
        self.train = True if splits.split('_')[1] == 'train' else False
        self.crop_img = crop_img

    def __len__(self):
        return self.lenids
    
    def __getitem__(self, idx):
        obj_id = self.ids[idx]
        if self.train:
            focal, H, W, imgs, poses, instances = self.return_train_data(obj_id)
            return focal, H, W, imgs, poses, instances, idx
        else:
            focal, H, W, imgs, poses = self.return_test_val_data(obj_id)
            return focal, H, W, imgs, poses, idx
    
    def return_train_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        instances = np.random.choice(50, self.num_instances_per_obj)
        poses = load_poses(pose_dir, instances)
        imgs = load_imgs(img_dir, instances)
        focal, H, W = load_intrinsic(intrinsic_path)
        if self.crop_img:
            imgs = imgs[:,32:-32,32:-32,:]
            H, W = H // 2, W//2
        return focal, H, W, imgs.reshape(self.num_instances_per_obj, -1,3), poses, instances
    
    def return_test_val_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        instances = np.arange(250)
        poses = load_poses(pose_dir, instances)
        imgs = load_imgs(img_dir, instances)
        focal, H, W = load_intrinsic(intrinsic_path)
        return focal, H, W, imgs, poses


if __name__ == '__main__':
    tgt_category = 'vehicle.car'
    cityscape_cat = 'car'
    divisor = 1000
    panoptic_folder = '/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/panoptic_pred'
    nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini', verbose=True)
    nusc.list_scenes()
    instance_all = nusc.instance
    tgt_instance_list = []
    # retrieve all the target instance
    for instance in instance_all:
        if nusc.get('category', instance['category_token'])['name'] == tgt_category:
            tgt_instance_list.append(instance)

    # retrieve all the annotations for each instance
    for instance in tgt_instance_list:
        ann_tokens = nusc.field2token('sample_annotation', 'instance_token', instance['token'])
        instoken=instance['token']
        print(f'instance: {instoken}')

        # For each annotation get all the sensors
        for anntoken in ann_tokens:
            print(f'    anntoken: {anntoken}')
            sample_ann = nusc.get('sample_annotation', anntoken)
            # nusc.render_annotation(anntoken)
            # plt.show()
            # plt.waitforbuttonpress(0)

            sample_record = nusc.get('sample', sample_ann['sample_token'])
            assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

            # Figure out which camera the object is fully visible in (this may return nothing).
            boxes, cam = [], []
            cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
            tgt_cams = []
            for cam in cams:
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_record['data'][cam],
                                                                          box_vis_level=BoxVisibility.ALL,
                                                                          selected_anntokens=[anntoken])
                if len(boxes) > 0:
                    assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
                                               'Try using e.g. BoxVisibility.ANY.'
                    assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

                    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
                    tgt_cams.append(cam)
                    camtoken = sample_record['data'][cam]

                    # Plot CAMERA view.
                    # data_path, boxes, camera_intrinsic = nusc.get_sample_data(camtoken, selected_anntokens=[anntoken])
                    im = Image.open(data_path)
                    axes[0].imshow(im)
                    axes[0].set_title(nusc.get('sample_data', camtoken)['channel'])
                    axes[0].axis('off')
                    axes[0].set_aspect('equal')
                    for box in boxes:
                        c = np.array(nusc.colormap[box.name]) / 255.0
                        box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c))

                    # visualize pred instance mask
                    pan_file = os.path.join(panoptic_folder, cam, 'panoptic', os.path.basename(data_path)[:-4]+'.png')
                    pan_img = np.asarray(Image.open(pan_file))
                    pan_label = img2pan(pan_img)
                    ins_vis = pan2ins_vis(pan_label, name2label[cityscape_cat][2], divisor)

                    axes[1].imshow(ins_vis)
                    axes[1].set_title('pred instance')
                    axes[1].axis('off')
                    axes[1].set_aspect('equal')
                    for box in boxes:
                        c = np.array(nusc.colormap[box.name]) / 255.0
                        corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
                        min_x = np.min(corners[0, :])
                        max_x = np.max(corners[0, :])
                        min_y = np.min(corners[1, :])
                        max_y = np.max(corners[1, :])
                        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                                 linewidth=2, edgecolor='y', facecolor='none')
                        axes[1].add_patch(rect)
                        tgt_ins_id, tgt_ins_cnt = tgt_instance(pan_label, name2label[cityscape_cat][2],
                                                               [min_x, min_y, max_x, max_y])
                        if tgt_ins_id is not None:
                            area_ratio = float(tgt_ins_cnt) / (max_x - min_x) / (max_y - min_y)
                            print(f'        tgt instand id: {tgt_ins_id}, num of pixels: {tgt_ins_cnt}, area ratio: {area_ratio}')
                        else:
                            print(f'        no tgt instance found')

                        # box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c))
                    plt.show()  # Press Q to quite the figure
                    # plt.waitforbuttonpress()
                    # plt.close('all')
