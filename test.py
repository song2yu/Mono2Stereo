import os
import cv2
import re
import numpy as np
from datetime import datetime
from PIL import Image, ImageOps, ImageChops
import torch
import matplotlib.pyplot as plt
from torch.nn import Conv2d
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils_depth.optimizer import build_optimizers
from util.loss import get_loss
import utils_depth.metrics as metrics
from torch.nn.parameter import Parameter
import utils_depth.logging as logging
from torchvision.transforms.functional import pil_to_tensor, resize
from torchvision.transforms import Compose
from config.test_options import TestOptions
from diffusers import DDPMScheduler
from util.multi_res_noise import multi_res_noise_like
import glob
import utils
from util.seeding import generate_seed_sequence
from marigold.marigold_pipeline import MarigoldPipeline
from depth.depth_anything_v2.dpt import DepthAnythingV2
from stereo.stereo_pipeline import Stereo
from util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from depth.depth_anything_v2.util.transform import MyResize, NormalizeImage, PrepareForNet


metric_name = ['rmse', 'mse', 'siou', 'psnr', 'ssim']


def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')

    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)
    logging.info("Unet weights are loaded")
    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)
        
    if ckpt_dict.get('best_mse'):
        return ckpt_dict['best_mse']


def _replace_unet_conv_in_12(painter, args):
    # replace the first layer to accept 8 in_channels
    _weight = painter.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = painter.unet.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, 3, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight *= 0.33333 
    # new conv_in channel
    _n_convin_out_channel = painter.unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        12, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    painter.unet.conv_in = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    # replace config
    painter.unet.config["in_channels"] = 12
    logging.info("Unet config is updated")

    return painter


def diff(pred, right):
    # convert to Pseudocolor 
    right = Image.fromarray(right).convert('L')
    pred = Image.fromarray(pred).convert('L')
    diff = ImageChops.difference(pred, right)
    diff_np = np.array(diff)
    diff_pseudocolor = apply_colormap(diff_np, 'hot')  # 'hot', 'jet', 'autumn'

    return diff_pseudocolor


def apply_colormap(gray_image, colormap_name):
    # Converting a Grayscale Image to a Pseudocolor Image
    colormap = plt.get_cmap(colormap_name)
    colored_image = colormap(gray_image / 255.0)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(colored_image)    


def main():
    # --------------------------------------------------configs initializing--------------------------------------------------------------------
    opt = TestOptions()
    args = opt.initialize().parse_args()
    # utils.init_distributed_mode_simple(args)
    print(args)
    if torch.cuda.is_available():
        args.gpu = 'cuda'
    else:
        args.gpu = 'cpu'
    device = torch.device(args.gpu)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    cfg = recursive_load_config(args.config)
    args.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
        os.path.join(
            args.base_ckpt_dir,
            cfg.trainer.training_noise_scheduler.pretrained_path,
            "scheduler",
        )
    )

    # ---------------------------------------------------loading model----------------------------------------------------------------------------
    depth_model = DepthAnythingV2(**model_configs[args.encoder])
    depth_model.load_state_dict(torch.load(f'depth/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu')) # Stereo/depth/checkpoint
    depth_model = depth_model.to(device).eval()
    _pipeline_kwargs = {'scale_invariant': True, 'shift_invariant': True}
    painter = MarigoldPipeline.from_pretrained(os.path.join("stabilityai/stable-diffusion-2"), **_pipeline_kwargs).to(device)
    if 12 != painter.unet.config["in_channels"]:
        painter = _replace_unet_conv_in_12(painter, args)
    
    cudnn.benchmark = True
    model = Stereo(depth_model=depth_model, painter=painter, args=args).to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu], find_unused_parameters=False)
    cfg.best_mse = load_model(args.weights, model.painter.unet)
    model.eval()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0


    args.cfg = cfg
    # ----------------------------------------------------testing---------------------------------------------------------------------------------------
    val_init_seed = args.cfg.validation.init_seed
    val_seed_ls = generate_seed_sequence(val_init_seed, length=1)
    seed = val_seed_ls.pop()

    transform = Compose([
            MyResize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    iters = 0
    file_path = args.out_dir
    next_guidance = None
    left_path = args.file_path
    new_size = (1280, 800)
    left_image = Image.open(args.file_path).convert("RGB").resize(new_size)
    right_path = args.file_path.replace('left', 'right')
    right_image = np.array(Image.open(right_path).convert("RGB").resize(new_size), dtype=np.uint8)
    rgb_left_norm = pil_to_tensor(left_image).unsqueeze(0)
    rgb_left_norm: torch.Tensor = (rgb_left_norm / 255.0 * 2.0 - 1.0).to(device)

    # convert to torch tensor [H, W, rgb] -> [rgb, H, W]    
    rgb_for_depth = cv2.imread(left_path)
    rgb_for_depth = cv2.cvtColor(rgb_for_depth, cv2.COLOR_BGR2RGB)
    rgb_for_depth = transform({'image': rgb_for_depth.astype(int)/255.0})['image']
    rgb_for_depth = torch.from_numpy(rgb_for_depth).unsqueeze(0).to(device)

    with torch.no_grad():
        viewpoint_latent = model.generate_viewpoint(rgb_left_norm, rgb_for_depth)
        pipe_out = model.painter(
            left_image,
            viewpoint_latent,
            denoising_steps=args.cfg.validation.denoising_steps,
            ensemble_size=args.cfg.validation.ensemble_size,
            processing_res=args.cfg.validation.processing_res,
            match_input_res=args.cfg.validation.match_input_res,
            generator=generator,
            batch_size=1,  
            color_map=None,
            show_progress_bar=False,
            resample_method=args.cfg.validation.resample_method,
        )

        right_pred: np.ndarray = pipe_out.right_pred
        right_to_save = pipe_out.right_visual
        keep_info = np.array(left_image)
        inpainted_pred = keep_info
        # get inpainting mask
        keep_mask = abs(keep_info - np.array(right_image)) > 5
        inpainted_pred[keep_mask] = right_to_save[keep_mask]
        left_for_test = np.array(left_image, dtype=np.uint8)
        # compute metrics
        computed_result = metrics.eval_stereo(right_to_save, right_image, left_for_test)
        print(computed_result)

        anaglyph_array = np.zeros_like(left_for_test)
        # extract red channel
        anaglyph_array[:, :, 0] = left_for_test[:, :, 0]

        # extract blue and green
        anaglyph_array[:, :, 1] = right_to_save[:, :, 1]
        anaglyph_array[:, :, 2] = right_to_save[:, :, 2]

        # convert to image
        anaglyph_image = Image.fromarray(anaglyph_array)
        anaglyph_image.save('ele.jpg')
        
 

if __name__ == '__main__':
    main()
