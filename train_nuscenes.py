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
    arg_parser.add_argument("--batch_size", dest="batch_size", type=int, default=3)
    arg_parser.add_argument("--ray_samples", dest="ray_samples", type=int, default=1600)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=4)
    arg_parser.add_argument("--iters_all", dest="iters_all", default=1200000)

    args = arg_parser.parse_args()

    nusc_seg_dir = os.path.join(args.nusc_data_dir, 'pred_' + args.seg_source)
    save_dir = args.nusc_cat + '.use_'+args.seg_source

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
        debug=False)

    trainer = TrainerNuScenes(save_dir, args.gpu, nusc_dataset,
                              args.pretrained_model_dir, args.jsonfile, args.batch_size,
                              args.ray_samples, num_workers=args.num_workers, shuffle=True)
    trainer.training(args.iters_all)
