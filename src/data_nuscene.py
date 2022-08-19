
import imageio
import numpy as np
import torch
# import json
# from torchvision import transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

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
            print(f'    qanntoken: {anntoken}')
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
                _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=BoxVisibility.ALL,
                                                   selected_anntokens=[anntoken])
                if len(boxes) > 0:

                    assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
                                               'Try using e.g. BoxVisibility.ANY.'
                    assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

                    fig, axes = plt.subplots(figsize=(9, 9))
                    tgt_cams.append(cam)
                    camtoken = sample_record['data'][cam]

                    # Plot CAMERA view.
                    data_path, boxes, camera_intrinsic = nusc.get_sample_data(camtoken, selected_anntokens=[anntoken])
                    im = Image.open(data_path)
                    axes.imshow(im)
                    axes.set_title(nusc.get('sample_data', camtoken)['channel'])
                    axes.axis('off')
                    axes.set_aspect('equal')
                    for box in boxes:
                        c = np.array(nusc.colormap[box.name]) / 255.0
                        box.render(axes, view=camera_intrinsic, normalize=True, colors=(c, c, c))

                    plt.show()
                    # plt.waitforbuttonpress()
                    # plt.close('all')
