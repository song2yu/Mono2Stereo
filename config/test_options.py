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
            default='/workspace/Mono2Stereo/checkpoint/mono2stereo.ckpt')
        parser.add_argument(
            "--file_path",
            type=str,
            default='images/left/000010938.jpg', 
        )
        parser.add_argument(
            "--base_path",
            type=str,
            default='base_path',
        )
        parser.add_argument(
            "--out_dir",
            type=str,
            default='out_dir',
        )

        return parser


