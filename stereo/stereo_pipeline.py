import logging
from typing import Dict, Optional, Union
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from marigold.marigold_pipeline import MarigoldPipeline
from depth.depth_anything_v2.dpt import DepthAnythingV2
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageChops

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

    
class Stereo(nn.Module):
    def __init__(
        self,
        depth_model: DepthAnythingV2,
        painter: MarigoldPipeline,
        args
    ):
        super().__init__()
        self.depth_model = depth_model
        self.painter = painter
        self.args = args
    
    @torch.no_grad()
    def pred_depth(self, left_image):
        depth = self.depth_model.infer_image(left_image)
        return depth

    @torch.no_grad()
    def generate_viewpoint(self, left_image, rgb_for_depth, baseline=8, focal_length=0.6):

        device = left_image.device
        b, c, h, w = left_image.shape
        depth_map = self.pred_depth(rgb_for_depth)
        if depth_map.shape[-2] != left_image.shape[-2]:
            depth_map = F.interpolate(depth_map.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
        right_image = torch.zeros_like(left_image, device=device)
        
        # compute disparity
        disparity_map = (baseline * focal_length) * depth_map  # (b, h, w)
        
        # coordinates
        x_coords = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w)
        right_x_coords = (x_coords - disparity_map).to(torch.int32)  
        
        valid_mask = (right_x_coords >= 0) & (right_x_coords < w)
        batch_indices = torch.arange(b, device=device).view(b, 1, 1).expand(b, h, w)
        channel_indices = torch.arange(c, device=device).view(1, c, 1, 1).expand(b, c, h, w)
        row_indices = torch.arange(h, device=device).view(1, 1, h, 1).expand(b, c, h, w)
        
        # index
        right_image[batch_indices.unsqueeze(1).expand(b, c, h, w), 
                    channel_indices, 
                    row_indices, 
                    right_x_coords.unsqueeze(1).expand(b, c, h, w)] = left_image[batch_indices.unsqueeze(1).expand(b, c, h, w), 
                                                                                channel_indices, 
                                                                                row_indices, 
                                                                                x_coords.unsqueeze(1).expand(b, c, h, w)]
    
        viewpoint_latent = self.painter.encode_rgb(right_image)
        if False:      
            right_image = right_image.squeeze()
            warp_image = right_image.cpu().numpy()
            warp_image = np.clip(warp_image, a_min=-1, a_max=1)
            warp_image = (warp_image+1) / 2.0 * 255
            warp_image = warp_image.astype(np.uint8)
            warp_image = chw2hwc(warp_image)

            return viewpoint_latent, warp_image
        
        else:
            return viewpoint_latent

