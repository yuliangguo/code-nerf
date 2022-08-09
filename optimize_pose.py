
import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse
from src.utils import str2bool
from src.optimizer import Optimizer


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu", dest="gpu", default=0)
    arg_parser.add_argument("--saved_dir", dest="saved_dir", default='srncar')
    arg_parser.add_argument("--tgt_instances", dest="tgt_instances", nargs='+', default=[0, 120],
                            help="the ids of the instances used for optimization, evaluation applies on the rest")
    arg_parser.add_argument("--splits", dest="splits", default='test')
    arg_parser.add_argument("--num_opts", dest="num_opts", default=200)
    arg_parser.add_argument("--lr", dest="lr", default=1e-2)
    arg_parser.add_argument("--lr_half_interval", dest="lr_half_interval", default=50)
    arg_parser.add_argument("--max_rot_pert", dest="max_rot_pert", default=0.1,
                            help="the max rotation perturbation applying to object pose")
    arg_parser.add_argument("--max_t_pert", dest="max_t_pert", default=0.1,
                            help="the max translation perturbation applying to object pose")
    arg_parser.add_argument("--save_img", dest="save_img", default=False)
    arg_parser.add_argument("--eval_pose_only", dest="eval_pose_only", default=True)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="srncar.json")
    arg_parser.add_argument("--batchsize", dest="batchsize", default=4096)

    args = arg_parser.parse_args()
    saved_dir = args.saved_dir
    gpu = int(args.gpu)
    lr = float(args.lr)
    lr_half_interval = int(args.lr_half_interval)
    save_img = str2bool(args.save_img)
    batchsize = int(args.batchsize)
    tgt_instances = list(args.tgt_instances)
    num_opts = int(args.num_opts)
    for num, i in enumerate(tgt_instances):
        tgt_instances[num] = int(i)
    save_postfix = f'_rpert{args.max_rot_pert}_tpert{args.max_t_pert}_nops{num_opts}_nview{len(tgt_instances)}'
    optimizer = Optimizer(saved_dir, gpu, tgt_instances, args.splits, args.jsonfile, batchsize, num_opts,
                          args.max_rot_pert, args.max_t_pert, save_postfix)
    optimizer.optimize_objs_w_pose(tgt_instances, lr, lr_half_interval, save_img, eval_pose_only=args.eval_pose_only)
