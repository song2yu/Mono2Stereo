# Author: Bingxin Ke
# Last modified: 2024-02-22

import torch
import torch.nn.functional as F

def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif "dis_loss" == loss_name:
        criterion = DisparityLoss(**kwargs)
    elif "mix_dis_loss" == loss_name:
        criterion = MixDisparityLoss(**kwargs)
    else:
        raise NotImplementedError

    return criterion

class MixDisparityLoss:
    def __init__(self, **kwargs):
        # Sobel
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # channel number
        self.sobel_x = self.sobel_x.repeat(4,1,1,1)
        self.sobel_y = self.sobel_y.repeat(4,1,1,1)
        self.mse = torch.nn.MSELoss(**kwargs)

    def __call__(self, pred, target, pred_latent, target_latent):
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)

        pred_grad = self.compute_gradient(pred_latent)
        target_grad = self.compute_gradient(target_latent)
        
        loss_dis = self.mse(pred_grad, target_grad) # 4.65

        loss = 0.01 * loss_dis + self.mse(pred, target)

        return loss

    def compute_gradient(self, feature_map):
        grad_x = F.conv2d(feature_map, self.sobel_x, padding=1, groups=4)
        grad_y = F.conv2d(feature_map, self.sobel_y, padding=1, groups=4)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        return gradient_magnitude

def compute_gradient(feature_map):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    

    sobel_x = sobel_x.repeat(4,1,1,1)
    sobel_y = sobel_y.repeat(4,1,1,1)

    sobel_x = sobel_x.to(feature_map.device)
    sobel_y = sobel_y.to(feature_map.device)

    grad_x = F.conv2d(feature_map, sobel_x, padding=1, groups=4)
    grad_y = F.conv2d(feature_map, sobel_y, padding=1, groups=4)

    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    return gradient_magnitude



class DisparityLoss:
    def __init__(self, **kwargs):
        # sobel
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x = self.sobel_x.repeat(1,4,1,1)
        self.sobel_y = self.sobel_y.repeat(1,4,1,1)

        self.mse = torch.nn.MSELoss(**kwargs)

    def __call__(self, pred, target):
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)

        pred_grad = self.compute_gradient(pred)
        target_grad = self.compute_gradient(target)
        
        loss = self.mse(pred_grad, target_grad)

        return loss


    def compute_gradient(self, feature_map):
        grad_x = F.conv2d(feature_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(feature_map, self.sobel_y, padding=1)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        return gradient_magnitude




class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss
