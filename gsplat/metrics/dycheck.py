from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "psnr",
    "mpsnr",
    "ssim",
    "mssim",
    "lpips",
    "mlpips",
]

_LPIPS_MODELS: Dict[Tuple[str, str], torch.nn.Module] = {}


def _to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tensor.shape}")
    if tensor.shape[1] in (1, 3, 4):
        return tensor
    if tensor.shape[-1] in (1, 3, 4):
        return tensor.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Cannot infer channel dimension for shape {tensor.shape}")


def _to_mask(mask: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected 4D mask, got shape {mask.shape}")
    if mask.shape[1] == 1:
        mask_bchw = mask
    elif mask.shape[-1] == 1:
        mask_bchw = mask.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Mask must have singleton channel dimension, got {mask.shape}")
    if mask_bchw.shape[2] != height or mask_bchw.shape[3] != width:
        mask_bchw = F.interpolate(mask_bchw, size=(height, width), mode="nearest")
    return (mask_bchw > 0.5).to(device=device, dtype=torch.float32)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = _to_bchw(pred)
    gt = _to_bchw(gt).to(pred.device)
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    psnr_vals = -10.0 * torch.log10(mse.clamp(min=1e-10))
    return psnr_vals.mean()


def mpsnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = _to_bchw(pred)
    gt = _to_bchw(gt).to(pred.device)
    mask_bchw = _to_mask(mask, pred.shape[2], pred.shape[3], pred.device)
    if mask_bchw.shape[0] != pred.shape[0]:
        raise ValueError("Mask batch size must match predictions.")
    valid = mask_bchw.sum(dim=(1, 2, 3))
    denom = valid * pred.shape[1]
    expanded_mask = mask_bchw.expand(-1, pred.shape[1], -1, -1)
    mse = ((pred - gt) ** 2 * expanded_mask).sum(dim=(1, 2, 3)) / denom.clamp(min=1e-10)
    psnr_vals = torch.where(
        valid > 1e-5,
        -10.0 * torch.log10(mse.clamp(min=1e-10)),
        torch.full_like(mse, float("inf")),
    )
    finite = torch.isfinite(psnr_vals)
    if not finite.any():
        return torch.tensor(float("inf"), device=pred.device)
    return psnr_vals[finite].mean()


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= kernel_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return kernel.repeat(channels, 1, 1, 1)


def _conv(img: torch.Tensor, kernel: torch.Tensor, padding: int) -> torch.Tensor:
    return F.conv2d(img, kernel, padding=padding, groups=img.shape[1])


def mssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    filter_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    min_coverage: float = 0.5,
) -> torch.Tensor:
    pred = _to_bchw(pred)
    gt = _to_bchw(gt).to(pred.device)
    mask_bchw = _to_mask(mask, pred.shape[2], pred.shape[3], pred.device)
    channels = pred.shape[1]
    kernel = _gaussian_kernel(filter_size, sigma, channels, pred.device)
    pad = filter_size // 2
    expanded_mask = mask_bchw.expand(-1, channels, -1, -1)
    coverage = _conv(expanded_mask, kernel, pad)
    valid = coverage >= min_coverage
    coverage = coverage.clamp_min(1e-6)
    pred_masked = pred * expanded_mask
    gt_masked = gt * expanded_mask
    mu_pred = _conv(pred_masked, kernel, pad) / coverage
    mu_gt = _conv(gt_masked, kernel, pad) / coverage
    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_pred_gt = mu_pred * mu_gt
    sigma_pred_sq = _conv(pred_masked ** 2, kernel, pad) / coverage - mu_pred_sq
    sigma_gt_sq = _conv(gt_masked ** 2, kernel, pad) / coverage - mu_gt_sq
    sigma_pred_gt = _conv(pred_masked * gt_masked, kernel, pad) / coverage - mu_pred_gt
    C1 = (k1 * 1.0) ** 2
    C2 = (k2 * 1.0) ** 2
    ssim_map = ((2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C2)) / (
        (mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2)
    )
    valid_weights = valid.float()
    weight_sum = valid_weights.sum()
    if weight_sum < 1e-5:
        return torch.tensor(0.0, device=pred.device)
    return (ssim_map * valid_weights).sum() / weight_sum


def ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    filter_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    ones_mask = torch.ones(pred.shape[0], 1, pred.shape[-2], pred.shape[-1], device=pred.device)
    return mssim(pred, gt, ones_mask, filter_size=filter_size, sigma=sigma, k1=k1, k2=k2, min_coverage=1.0)


def _get_lpips_model(net: str, device: torch.device):
    key = (net, f"{device.type}:{device.index if device.index is not None else 0}")
    model = _LPIPS_MODELS.get(key)
    if model is None:
        import lpips  # type: ignore

        model = lpips.LPIPS(net=net, spatial=True)
        model = model.to(device)
        model.eval()
        _LPIPS_MODELS[key] = model
    return model


def lpips(pred: torch.Tensor, gt: torch.Tensor, net: str = "alex") -> torch.Tensor:
    pred = _to_bchw(pred)
    gt = _to_bchw(gt).to(pred.device)
    mask = torch.ones(pred.shape[0], 1, pred.shape[2], pred.shape[3], device=pred.device)
    return mlpips(pred, gt, mask, net=net)


def mlpips(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    net: str = "alex",
) -> torch.Tensor:
    pred = _to_bchw(pred)
    gt = _to_bchw(gt).to(pred.device)
    mask_bchw = _to_mask(mask, pred.shape[2], pred.shape[3], pred.device)
    valid = mask_bchw.sum(dim=(1, 2, 3))
    if not torch.any(valid > 1e-5):
        return torch.tensor(0.0, device=pred.device)
    device = pred.device
    model = _get_lpips_model(net, device)
    pred_lp = pred * 2.0 - 1.0
    gt_lp = gt * 2.0 - 1.0
    expanded_mask = mask_bchw.expand_as(pred_lp)
    with torch.no_grad():
        dist_map = model(pred_lp * expanded_mask, gt_lp * expanded_mask)
    mask_down = F.interpolate(mask_bchw, size=dist_map.shape[2:], mode="nearest")
    valid_sum = mask_down.sum()
    if valid_sum < 1e-5:
        return torch.tensor(0.0, device=device)
    return (dist_map * mask_down).sum() / valid_sum
