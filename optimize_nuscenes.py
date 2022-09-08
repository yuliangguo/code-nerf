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
    arg_parser.add_argument("--cs_cat", dest="cs_cat", default='car',
                            help="cityscape category name")
    arg_parser.add_argument("--nusc_data_dir", dest="nusc_data_dir",
                            default='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini',
                            help="nuscenes dataset dir")
    arg_parser.add_argument("--nusc_pan_dir", dest="nusc_pan_dir",
                            default='/mnt/LinuxDataFast/Datasets/NuScenes/v1.0-mini/panoptic_pred',
                            help="predicted panoptic segmentation onnuscenes dataset")
    arg_parser.add_argument("--nvsc_version", dest="nvsc_version", default='v1.0-mini',
                            help="version number required to load nuscene ground-truh")
    arg_parser.add_argument("--num_cams_per_sample", dest="num_cams_per_sample", type=int, default=1)
    arg_parser.add_argument("--num_opts", dest="num_opts", type=int, default=10)  # Early overfit for single image
    arg_parser.add_argument("--lr", dest="lr", type=float, default=1e-2)
    arg_parser.add_argument("--lr_half_interval", dest="lr_half_interval", type=int, default=50)
    arg_parser.add_argument("--save_img", dest="save_img", default=True)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="srncar.json")
    arg_parser.add_argument("--batchsize", dest="batchsize", type=int, default=1800)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)

    args = arg_parser.parse_args()

    nusc_dataset = NuScenesData(
        nusc_cat=args.nusc_cat,
        cs_cat=args.cs_cat,
        nusc_data_dir=args.nusc_data_dir,
        nusc_pan_dir=args.nusc_pan_dir,
        nvsc_version=args.nvsc_version,
        num_cams_per_sample=args.num_cams_per_sample,
        divisor=1000,
        box_iou_th=0.5,
        mask_pixels=10000,
        img_h=900,
        img_w=1600,
        debug=False)

    save_postfix = '_nuscenes_multi'
    optimizer = OptimizerNuScenes(args.model_dir, args.gpu, nusc_dataset, args.jsonfile,
                                  args.batchsize, args.num_opts, args.num_cams_per_sample,
                                  num_workers=args.num_workers, shuffle=False, save_postfix=save_postfix)
    # optimizer.optimize_objs(args.lr, args.lr_half_interval, str2bool(args.save_img))
    optimizer.optimize_objs_multi_anns(args.lr, args.lr_half_interval, str2bool(args.save_img))
