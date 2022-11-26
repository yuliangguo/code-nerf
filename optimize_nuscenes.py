import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse

from src.utils import str2bool
from src.optimizer_codenerf_nuscenes import OptimizerNuScenes
from src.optimizer_autorf_nuscenes import OptimizerAutoRFNuScenes
from src.data_nuscenes import NuScenesData


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    arg_parser.add_argument("--config_file", dest="config_file", default="autorf.nusc.vehicle.car.json")
    arg_parser.add_argument("--model_dir", dest="model_dir", default='exps_nuscenes_autorf/vehicle.car.v1.0-trainval.use_instance.bsize10_2022_11_23',
                            help="location of saved pretrained model and codes")
    arg_parser.add_argument("--nusc_cat", dest="nusc_cat", default='vehicle.car',
                            help="nuscence category name")
    arg_parser.add_argument("--seg_cat", dest="seg_cat", default='car',
                            help="predicted segment category name")
    arg_parser.add_argument("--nusc_data_dir", dest="nusc_data_dir",
                            default='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini',
                            help="nuscenes dataset dir")
    arg_parser.add_argument("--nusc_version", dest="nusc_version", default='v1.0-mini',
                            help="version number required to load nuscene ground-truth")
    arg_parser.add_argument("--seg_source", dest="seg_source",
                            default='instance',
                            help="use predicted instance/panoptic segmentation on nuscenes dataset")
    arg_parser.add_argument("--save_img", dest="save_img", default=True)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)
    arg_parser.add_argument("--multi_ann_ops", dest="multi_ann_ops", default=True,
                            help="if to optimize multiple annotations of the same instance jointly")
    arg_parser.add_argument("--opt_pose", dest="opt_pose", default=True,
                            help="if to optimize camera poses, if true the dataloader will generate erroneous poses")
    arg_parser.add_argument("--rot_err", dest="rot_err", default=0.2,
                            help='add rotation error in rad to data loading when optimizing pose')
    arg_parser.add_argument("--t_err", dest="t_err", default=0.001,
                            help='add translation error in ratio added in data loading when optimizing pose')

    args = arg_parser.parse_args()

    nusc_seg_dir = os.path.join(args.nusc_data_dir, 'pred_' + args.seg_source)
    save_postfix = '_nuscenes_use_' + args.seg_source
    if args.multi_ann_ops:
        save_postfix += '_multi_ann'
    if args.opt_pose:
        save_postfix += '_opt_pose'

    nusc_dataset = NuScenesData(
        nusc_cat=args.nusc_cat,
        seg_cat=args.seg_cat,
        nusc_data_dir=args.nusc_data_dir,
        nusc_seg_dir=nusc_seg_dir,
        nusc_version=args.nusc_version,
        split='val',
        num_cams_per_sample=1,
        divisor=1000,
        box_iou_th=0.5,
        mask_pixels=2500,
        img_h=900,
        img_w=1600,
        debug=False,
        add_pose_err=args.opt_pose,
        max_rot_pert=args.rot_err,
        max_t_pert=args.t_err,
    )

    if 'autorf' in args.config_file:
        optimizer = OptimizerAutoRFNuScenes(args.model_dir, args.gpu, nusc_dataset, args.config_file,
                                            num_workers=args.num_workers, shuffle=False, save_postfix=save_postfix)
    else:
        optimizer = OptimizerNuScenes(args.model_dir, args.gpu, nusc_dataset, args.config_file,
                                      num_workers=args.num_workers, shuffle=False, save_postfix=save_postfix)

    if args.opt_pose:
        if args.multi_ann_ops:
            optimizer.optimize_objs_multi_anns_w_pose(str2bool(args.save_img))
        else:
            optimizer.optimize_objs_w_pose(str2bool(args.save_img))
    else:
        if args.multi_ann_ops:
            optimizer.optimize_objs_multi_anns(str2bool(args.save_img))
        else:
            optimizer.optimize_objs(str2bool(args.save_img))
