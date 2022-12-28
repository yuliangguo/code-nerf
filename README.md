# NeRF Reconstruction for Autonomous Driving

## Install the environment

The framework builds upon previous works including 
[CodeNeRF](https://arxiv.org/abs/2109.01750) and [AutoRF](https://arxiv.org/abs/2204.03593). Training and testing pipelines have been implemented for [NuScenes](https://www.nuscenes.org/nuscenes) Dataset.

## Install the environment

```
conda create -y -n nerf-ad python=3.8
conda activate nerf-ad

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

## Catalog

- [x] Training
- [x] Optimizing with GT pose
- [x] Editing Shapes/Textures
- [x] Pose Optimizing


## Data preparation (NuScenes)

You can save the download NuScenes dataset to you local folder `Data_ROOT/NuScenes/`. The required data structure will look like

        NuScenes
        ├── v1.0-trainval
        │   ├── maps
        |   |       
        │   ├── samples
        |   |       
        │   ├── sweeps
        │   |   
        │   ├── v1.0-trainval
        |   |       
        │   ├── pred_instance
        |   |       
        │   ├── pred_panoptic
        │   |
        │   ├── pred_depth
        |── v1.0-mini
        ...

In addition to the original downloaded data, `pred_instance` is prepared via script in [mask-rcnn detectron2](https://github.com/yuliangguo/detectron2/tree/main/demo). 
If to use panoptic segmentation instead, `pred_panoptic` is prepared via script in [panoptic-deeplab](https://github.com/yuliangguo/panoptic-deeplab/tree/master/tools).
In case to use monocular depth prior, the `pred_depth` folder includes all the monocular depth maps is prepared via script in [MiDaS](https://github.com/yuliangguo/MiDaS).

## Training

To train a NeRF reconstruction network on NuScenes, we proved two different config files with codenerf and autorf architecture separately.
Before training, the `train_data_dir` and `train_nusc_version` need to be manually set based on your local dataset folder, then execute
```
python train_nuscenes.py --gpu <gpu_id> --config_file <config_file.json>
```
Other hyparparameters can be modified from `train_nuscenes.py` or the .json files saved in `jsonfiles/`.

The original training code for codenerf is renamed as `train_srn.py`. Please refer the codenerf readme to run it.


## Optimizing

For optimization of shaper, texture and pose on the testing set, you need to first assign the config_file as the one saved in the trained model folder,
e.g., `exps_nuscenes_autorf/vehicle.car.v1.0-trainval.use_instance.bsize6_2022_12_02/hpam.json`. There are different mode in the optimization, 
wt./wo pose refinement, single/multi view observations. There can be modified in `optimize_nuscenes.py`

```
python optimize_nuscenes.py --gpu <gpu_id> --config_file <config_file.json>
```

The result will be stored in <trained_dir/test(_num)>, and each folder contains the progress of optimization, and the evaluation of test set. 
The final optimized results and the quantitative evaluations are stored in `trained_dir/test(_num)/codes.pth`

The original optimization code for codenerf is renamed as `optimize_srn.py`. Please refer the codenerf readme to run it. In addition, pose refinement
is added to the original release of the codenerf code.

### License

MIT

