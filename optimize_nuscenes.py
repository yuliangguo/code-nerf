import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse

from src.utils import str2bool
from src.optimizer_nuscenes import OptimizerNuScenes
from src.data_nuscene import NuScenesData


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    arg_parser.add_argument("--model_dir", dest="model_dir", default='srncar_08182022',
                            help="location of saved pretrained model and codes")
    arg_parser.add_argument("--nusc_cat", dest="nusc_cat", default='vehicle.car',
                            help="nuscence category name")
    arg_parser.add_argument("--seg_cat", dest="seg_cat", default='car',
                            help="predicted segment category name")
    arg_parser.add_argument("--nusc_data_dir", dest="nusc_data_dir",
                            default='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini',
                            help="nuscenes dataset dir")
    arg_parser.add_argument("--seg_source", dest="seg_source",
                            default='instance',
                            help="use predicted instance/panoptic segmentation on nuscenes dataset")
    arg_parser.add_argument("--nusc_version", dest="nusc_version", default='v1.0-mini',
                            help="version number required to load nuscene ground-truh")
    # arg_parser.add_argument("--num_cams_per_sample", dest="num_cams_per_sample", type=int, default=1)
    arg_parser.add_argument("--num_opts", dest="num_opts", type=int, default=30)  # Early overfit for single image
    arg_parser.add_argument("--lr", dest="lr", type=float, default=1e-2)
    arg_parser.add_argument("--lr_half_interval", dest="lr_half_interval", type=int, default=10)
    arg_parser.add_argument("--save_img", dest="save_img", default=True)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="srncar.json")
    arg_parser.add_argument("--n_rays", dest="n_rays", type=int, default=1800)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)
    arg_parser.add_argument("--opt_pose", dest="opt_pose", default=True)

    args = arg_parser.parse_args()

    nusc_seg_dir = os.path.join(args.nusc_data_dir, 'pred_' + args.seg_source)
    save_postfix = '_nuscenes_use_' + args.seg_source
    if args.opt_pose:
        save_postfix += '_opt_pose'

    nusc_dataset = NuScenesData(
        nusc_cat=args.nusc_cat,
        seg_cat=args.seg_cat,
        nusc_data_dir=args.nusc_data_dir,
        nusc_seg_dir=nusc_seg_dir,
        nusc_version=args.nusc_version,
        num_cams_per_sample=1,
        divisor=1000,
        box_iou_th=0.5,
        mask_pixels=3000,
        img_h=900,
        img_w=1600,
        debug=False,
        add_pose_err=args.opt_pose)

    optimizer = OptimizerNuScenes(args.model_dir, args.gpu, nusc_dataset, args.jsonfile,
                                  args.n_rays, args.num_opts, num_cams_per_sample=1,
                                  num_workers=args.num_workers, shuffle=True, save_postfix=save_postfix)

    if args.opt_pose:
        optimizer.optimize_objs_w_pose(args.lr, args.lr_half_interval, str2bool(args.save_img))
    else:
        # optimizer.optimize_objs(args.lr, args.lr_half_interval, str2bool(args.save_img))
        optimizer.optimize_objs_multi_anns(args.lr, args.lr_half_interval, str2bool(args.save_img))
