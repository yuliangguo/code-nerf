import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse
from datetime import date

from src.trainer_codenerf_nuscenes import TrainerNuScenes
from src.trainer_autorf_nuscenes import TrainerAutoRFNuScenes
from src.data_nuscenes import NuScenesData


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    arg_parser.add_argument("--config_file", dest="config_file", default="autorf.nusc.vehicle.car.json")
    arg_parser.add_argument("--nusc_data_dir", dest="nusc_data_dir",
                            default='/media/yuliangguo/data_ssd_4tb/Datasets/nuscenes/v1.0-trainval',
                            help="nuscenes dataset dir")
    arg_parser.add_argument("--nusc_version", dest="nusc_version", default='v1.0-trainval',
                            help="version number required to load nuscene ground-truth")
    arg_parser.add_argument("--nusc_cat", dest="nusc_cat", default='vehicle.car',
                            help="nuscence category name")
    arg_parser.add_argument("--seg_cat", dest="seg_cat", default='car',
                            help="predicted segment category name")
    arg_parser.add_argument("--seg_source", dest="seg_source",
                            default='instance',
                            help="use predicted instance/panoptic segmentation on nuscenes dataset")
    arg_parser.add_argument("--pretrained_model_dir", dest="pretrained_model_dir", default=None,
                            help="location of saved pretrained model and codes")
    arg_parser.add_argument("--batch_size", dest="batch_size", type=int, default=10)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=4)
    arg_parser.add_argument("--epochs", dest="epochs", default=20)
    arg_parser.add_argument("--resume_from_epoch", dest="resume_from_epoch", default=None)

    args = arg_parser.parse_args()

    nusc_seg_dir = os.path.join(args.nusc_data_dir, 'pred_' + args.seg_source)
    today = date.today()
    dt_str = today.strftime('_%Y_%m_%d')
    save_dir = args.nusc_cat + '.' + args.nusc_version + '.use_' + args.seg_source + f'.bsize{args.batch_size}' + dt_str

    nusc_dataset = NuScenesData(
        nusc_cat=args.nusc_cat,
        seg_cat=args.seg_cat,
        nusc_data_dir=args.nusc_data_dir,
        nusc_seg_dir=nusc_seg_dir,
        nusc_version=args.nusc_version,
        split='train',
        num_cams_per_sample=1,
        divisor=1000,
        box_iou_th=0.5,
        mask_pixels=2500,
        img_h=900,
        img_w=1600,
        debug=False)

    if 'autorf' in args.config_file:
        trainer = TrainerAutoRFNuScenes(save_dir, args.gpu, nusc_dataset,
                                        args.pretrained_model_dir, args.resume_from_epoch, args.config_file,
                                        args.batch_size, num_workers=args.num_workers, shuffle=True)
    else:
        trainer = TrainerNuScenes(save_dir, args.gpu, nusc_dataset,
                                  args.pretrained_model_dir, args.resume_from_epoch, args.config_file, args.batch_size,
                                  num_workers=args.num_workers, shuffle=True)
    trainer.training(args.epochs)
