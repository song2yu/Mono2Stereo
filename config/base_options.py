# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--exp_name',   type=str, default='')
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')

        parser.add_argument('--dataset',      type=str, default='nyudepthv2',
                            choices=['nyudepthv2', 'kitti', 'mono2stereo'])
        parser.add_argument('--workers',      type=int, default=8)
        parser.add_argument('--base_data_dir',   type=str, default='/workspace/dataset/')
        parser.add_argument('--base_ckpt_dir',   type=str, default='checkpoint/')
        
        # depth configs
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
        parser.add_argument('--do_kb_crop',     type=int, default=1)
        parser.add_argument('--kitti_crop', type=str, default=None,
                            choices=['garg_crop', 'eigen_crop'])

        parser.add_argument('--pretrained',    type=str, default='')
        parser.add_argument('--use_checkpoint',   type=str2bool, default='False')
        parser.add_argument('--num_deconv',     type=int, default=3)

      
        
        return parser
