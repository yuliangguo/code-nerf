import sys, os
ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))
import argparse
import json
from datetime import date

# from src.trainer_nuscenes import TrainerNuScenes
from src.trainer_nuscenes import TrainerNuScenes
from src.data_nuscenes import NuScenesData


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gpus", dest="gpus", type=int, default=4,
                            help='Number of GPUs to use')
    arg_parser.add_argument("--config_file", dest="config_file", default="jsonfiles/autorf.nusc.vehicle.car.json")
    arg_parser.add_argument("--seg_source", dest="seg_source",
                            default='instance',
                            help="use predicted instance/panoptic segmentation on nuscenes dataset")
    arg_parser.add_argument("--pretrained_model_dir", dest="pretrained_model_dir", default=None,
                            help="location of saved pretrained model and codes")
    arg_parser.add_argument("--batch_size", dest="batch_size", type=int, default=24)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=16)
    arg_parser.add_argument("--epochs", dest="epochs", default=20)
    arg_parser.add_argument("--resume_from_epoch", dest="resume_from_epoch", default=None)
    args = arg_parser.parse_args()

    # Read Hyper-parameters
    with open(args.config_file, 'r') as f:
        hpams = json.load(f)

    nusc_data_dir = hpams['dataset']['train_data_dir']
    nusc_seg_dir = os.path.join(nusc_data_dir, 'pred_' + args.seg_source)
    nusc_version = hpams['dataset']['train_nusc_version']
    today = date.today()
    dt_str = today.strftime('_%Y_%m_%d')
    save_dir = hpams['dataset']['nusc_cat'] + '.' + nusc_version + '.use_' + args.seg_source + f'.bsize{args.batch_size}' + dt_str

    # create dataset
    nusc_dataset = NuScenesData(
        hpams,
        nusc_data_dir,
        nusc_seg_dir,
        nusc_version,
        split='train',
        is_train=True,
        debug=False,
    )

    # create trainer
    trainer = TrainerNuScenes(save_dir, args.gpus, nusc_dataset,
                              args.pretrained_model_dir, args.resume_from_epoch, args.config_file,
                              args.batch_size, num_workers=args.num_workers, shuffle=True)

    # training
    trainer.train(args.epochs)
