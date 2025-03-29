# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from config.base_options import BaseOptions
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


class TrainOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument(
        "--config",
        type=str,
        default="config/train_stereo.yaml",
        help="Path to config file.",
        )
        parser.add_argument('--epochs',      type=int,   default=10)
        parser.add_argument('--initial_lr',          type=float, default=1e-5)        
        parser.add_argument('--log_dir', type=str, default='/group/40034/brucessyu/MyStereo/log_dir/')
        parser.add_argument('--resume_run', type=str, default='/workdir/') 
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--accumulate_steps', type=int, default=1)
        # logging options
        parser.add_argument('--encoder', type=str, default='vits')
        parser.add_argument('--val_freq', type=int, default=100)
        parser.add_argument('--pro_bar', type=str2bool, default='False')
        parser.add_argument('--print_freq', type=int, default=10)
 
        return parser
