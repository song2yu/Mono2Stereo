# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

from config.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        parser.add_argument(
        "--config",
        type=str,
        default="config/train_stereo.yaml",
        help="Path to config file.",
        )
        # logging options
        parser.add_argument(
            '--encoder', 
            type=str, 
            default='vits')
        parser.add_argument(
            '--weights', 
            type=str, 
            default='/workspace/log_dir/24_10_18-17_30_26_inria_dataset_500/epoch_170_last_18500_.ckpt')
        parser.add_argument(
            "--file_path",
            type=str,
            default='images/sora.png', 
        )
        parser.add_argument(
            "--base_path",
            type=str,
            default='/group/40043/brucessyu/datasets/',
        )
        parser.add_argument(
            "--out_dir",
            type=str,
            default='/workspace/outputs/10_18-17dual/video/',
        )

        return parser


