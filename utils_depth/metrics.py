# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def detect_edges(image, low, high):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)  # 30 50
    return edges

def edge_overlap(edge1, edge2):
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    return intersection / union

    
def compute_siou(pred, target, left):
    left_edges = detect_edges(left, 100, 200)
    pred_edges = detect_edges(pred, 100, 200) # 5, 50
    right_edges = detect_edges(target, 100, 200) # 20 100


    diff_gl = abs(pred - left)
    diff_rl = abs(target - left)
    diff_gl = cv2.cvtColor(diff_gl, cv2.COLOR_BGR2GRAY)
    diff_rl = cv2.cvtColor(diff_rl, cv2.COLOR_BGR2GRAY)
    diff_gl_ = np.zeros(diff_rl.shape)
    diff_rl_ = np.zeros(diff_rl.shape)
    diff_gl_[diff_gl>5] = 1 # 5
    diff_rl_[diff_rl>5] = 1

    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    diff_overlap_grl =edge_overlap(diff_gl_, diff_rl_)


    return 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl



def eval_stereo(pred, target, left):
    max_pixel = 255.0
    assert pred.shape == target.shape
    diff = pred - target

    mse_err = np.mean(diff ** 2)

    rmse = np.sqrt(mse_err)

    absolute_errors = np.abs(diff)
    mae = np.mean(absolute_errors)
    
    psnr = 20 * np.log10(max_pixel / rmse)

    ssim_value, _ = ssim(pred, target, full=True, multichannel=True, win_size=7, channel_axis=2)
    siou_value = compute_siou(pred, target, left)

    return {'rmse': rmse.item(), 'mse':mse_err.item(), 'siou': siou_value.item(), 'psnr': psnr.item(), 'ssim': ssim_value.item()}




def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10':log10.item(), 'silog':silog.item()}


def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval
    
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if args.dataset == 'kitti':
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_depth = gt_depth[top_margin:top_margin +
                            352, left_margin:left_margin + 1216]            

        if args.kitti_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros(valid_mask.shape).to(
                device=valid_mask.device)

            if args.kitti_crop == 'garg_crop':
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.kitti_crop == 'eigen_crop':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask = valid_mask

    elif args.dataset == 'nyudepthv2':
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        eval_mask[45:471, 41:601] = 1
    else:
        eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]

