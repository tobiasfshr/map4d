from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def ssim(
    target_rgbs: torch.Tensor,
    rgbs: torch.Tensor,
    max_val: float = 1,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: str = "mean",
) -> Tensor:
    """Computes SSIM from two images.

    Modified from: https://github.com/hturki/suds/blob/main/suds/metrics.py

    Args:
        rgbs: torch.tensor. An image of size [..., width, height, num_channels].
        target_rgbs: torch.tensor. An image of size [..., width, height, num_channels].
        max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
        filter_size: int >= 1. Window size.
        filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
        k1: float > 0. One of the SSIM dampening parameters.
        k2: float > 0. One of the SSIM dampening parameters.
        reduction: str. One of ['mean', 'none']. The reduction method for the output.
    Returns:
        Each image's mean SSIM.
    """
    device = rgbs.device
    ori_shape = rgbs.size()
    width, height, num_channels = ori_shape[-3:]
    rgbs = rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    target_rgbs = target_rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    if reduction == "mean":
        pad1, pad2 = 0, 0
    else:
        pad1 = [hw, 0]
        pad2 = [0, hw]

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1), padding=pad1, groups=num_channels
    )
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1), padding=pad2, groups=num_channels
    )

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(rgbs)
    mu1 = filt_fn(target_rgbs)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgbs**2) - mu00
    sigma11 = filt_fn(target_rgbs**2) - mu11
    sigma01 = filt_fn(rgbs * target_rgbs) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    if reduction == "mean":
        return ssim_map.mean()
    assert reduction == "none", f"Invalid reduction method: {reduction}"
    return ssim_map


def depth_metrics(pred_depth: Tensor, gt_depth: Tensor) -> Dict[str, float]:
    """We compute the standard depth metrics and store them in a dictionary
    with the corresponding names.

    Args:
        pred_depth (Tensor): predicted depth map
        gt_depth (Tensor): ground truth depth map
    Returns:
        Dict[str, float]: depth metrics
    """
    metrics = {}
    mask = gt_depth > 0.0

    thresh = torch.max((gt_depth[mask] / pred_depth[mask]), (pred_depth[mask] / gt_depth[mask]))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    metrics = {"depth_a1": a1, "depth_a2": a2, "depth_a3": a3}

    rmse = (gt_depth[mask] - pred_depth[mask]) ** 2
    rmse = torch.sqrt(rmse.mean())
    metrics["depth_rmse"] = rmse

    rmse_log = (torch.log(gt_depth[mask]) - torch.log(pred_depth[mask])) ** 2
    rmse_log = torch.sqrt(rmse_log).nanmean()
    metrics["depth_rmse_log"] = rmse

    abs_rel = torch.abs(gt_depth - pred_depth)[mask] / gt_depth[mask]
    abs_rel = abs_rel.mean()
    metrics["depth_abs_rel"] = abs_rel

    sq_rel = (gt_depth - pred_depth)[mask] ** 2 / gt_depth[mask]
    sq_rel = sq_rel.mean()
    metrics["depth_sq_rel"] = sq_rel

    return metrics


def psnr(image_gt: Tensor, image_pred: Tensor, reduction: str = "mean", eps: float = 1e-8) -> Tensor:
    """Compute PSNR between two images.

    Args:
        image_gt (Tensor): ground truth image
        image_pred (Tensor): predicted image
        reduction (str, optional): Reduction type. Defaults to 'mean'.
        eps (float, optional): Small value to avoid log(0). Defaults to 1e-8.

    Returns:
        Tensor: PSNR value(s).
    """
    return -10 * torch.log10(F.mse_loss(image_pred, image_gt, reduction=reduction).clamp(min=eps))
