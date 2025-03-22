import cv2
import torch
import numpy as np
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('/workspace/outputs/warp_14-00_30_guidance_left/video/7/00020001.jpg')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = np.array(depth.cpu()).astype(np.uint8)
depth = cv2.resize(depth, (960, 540), interpolation=cv2.INTER_LINEAR)
#split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
#combined_result = cv2.hconcat([raw_img, split_region, depth])
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


cv2.imwrite('/workspace/MyStereo/depth/outputs/img_20001.png', depth)