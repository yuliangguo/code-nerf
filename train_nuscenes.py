import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse

from src.trainer_nuscenes import TrainerNuScenes
from src.data_nuscene import NuScenesData


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    arg_parser.add_argument("--pretrained_model_dir", dest="pretrained_model_dir", default='srncar_08182022',
                            help="location of saved pretrained model and codes")
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="srncar.json")
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
    arg_parser.add_argument("--batchsize", dest="batchsize", type=int, default=1800)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)
    arg_parser.add_argument("--iters_all", dest="iters_all", default=1200000)

    args = arg_parser.parse_args()

    nusc_dataset = NuScenesData(
        nusc_cat=args.nusc_cat,
        cs_cat=args.cs_cat,
        nusc_data_dir=args.nusc_data_dir,
        nusc_pan_dir=args.nusc_pan_dir,
        nvsc_version=args.nvsc_version,
        num_cams_per_sample=1,
        divisor=1000,
        box_iou_th=0.5,
        mask_pixels=2*args.batchsize,
        img_h=900,
        img_w=1600,
        debug=False)

    optimizer = TrainerNuScenes(args.nusc_cat, args.gpu, nusc_dataset,
                                args.pretrained_model_dir, args.jsonfile,
                                args.batchsize, num_workers=args.num_workers, shuffle=False)
    optimizer.training(args.iters_all)
