import sys, os
ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))
import argparse
import json
import numpy as np

from src.utils import str2bool
from src.optimizer_nuscenes import OptimizerNuScenes
from src.data_nuscenes import NuScenesData


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    arg_parser.add_argument("--config_file", dest="config_file", default="exps_nuscenes_autorf/vehicle.car.v1.0-trainval.use_instance.bsize6_2022_12_02/hpam.json")
    arg_parser.add_argument("--seg_source", dest="seg_source", default='instance',
                            help="use predicted instance/panoptic segmentation on nuscenes dataset")
    arg_parser.add_argument("--save_img", dest="save_img", default=False)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)
    arg_parser.add_argument("--multi_ann_ops", dest="multi_ann_ops", default=False,
                            help="if to optimize multiple annotations of the same instance jointly")
    arg_parser.add_argument("--opt_pose", dest="opt_pose", default=False,
                            help="if to optimize camera poses, if true the dataloader will generate erroneous poses")
    args = arg_parser.parse_args()

    # Read Hyper-parameters
    with open(args.config_file, 'r') as f:
        hpams = json.load(f)

    nusc_data_dir = hpams['dataset']['test_data_dir']
    nusc_seg_dir = os.path.join(nusc_data_dir, 'pred_' + args.seg_source)
    nusc_version = hpams['dataset']['test_nusc_version']

    # create dataset
    nusc_dataset = NuScenesData(
        hpams,
        nusc_data_dir,
        nusc_seg_dir,
        nusc_version,
        split='val',
        is_train=False,
        debug=False,
        add_pose_err=args.opt_pose
    )

    # create optimizer
    save_postfix = '_nuscenes_use_' + args.seg_source
    if args.multi_ann_ops:
        save_postfix += '_multi_ann'
    if args.opt_pose:
        save_postfix += '_opt_pose'
    optimizer = OptimizerNuScenes(args.gpu, nusc_dataset, hpams,
                                  num_workers=args.num_workers, shuffle=False, save_postfix=save_postfix)

    # run-time optimization
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

    # eval summary
    psnr_all = []
    for psnr in optimizer.psnr_eval.values():
        psnr_all.append(psnr[0])
    print(f'Avg psnr error: {np.mean(np.array(psnr_all))}')
    if args.opt_pose:
        R_err_all = []
        T_err_all = []
        for R_err in optimizer.R_eval.values():
            R_err_all.append(R_err.numpy())
        for T_err in optimizer.T_eval.values():
            T_err_all.append(T_err.numpy())
        print(f'Avg R error: {np.mean(np.array(R_err_all))}, Avg T error: {np.mean(np.array(T_err_all))}')
