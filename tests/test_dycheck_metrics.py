import torch

from gsplat.metrics import dycheck as dy_metrics


def test_psnr_matches_mpsnr_with_full_mask():
    pred = torch.rand(2, 3, 16, 16)
    gt = torch.rand(2, 3, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    psnr_full = dy_metrics.psnr(pred, gt)
    psnr_masked = dy_metrics.mpsnr(pred, gt, mask)
    assert torch.allclose(psnr_full, psnr_masked, atol=1e-5)


def test_mpsnr_improves_when_mask_excludes_errors():
    gt = torch.zeros(1, 3, 16, 16)
    pred = gt.clone()
    pred[:, :, 8:, :] = 1.0
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :8, :] = 1.0
    full = dy_metrics.psnr(pred, gt)
    masked = dy_metrics.mpsnr(pred, gt, mask)
    assert masked > full + 5.0  # substantial improvement with easier region


def test_mssim_accepts_bhwc_and_bchw():
    pred = torch.rand(1, 3, 32, 32)
    gt = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    bhwc_pred = pred.permute(0, 2, 3, 1)
    bhwc_gt = gt.permute(0, 2, 3, 1)
    bhwc_mask = mask.permute(0, 2, 3, 1)
    bchw_ssim = dy_metrics.mssim(pred, gt, mask)
    bhwc_ssim = dy_metrics.mssim(bhwc_pred, bhwc_gt, bhwc_mask)
    assert torch.allclose(bchw_ssim, bhwc_ssim, atol=1e-5)


def test_mlpips_identical_images_is_zero():
    pred = torch.rand(1, 3, 64, 64)
    mask = torch.ones(1, 1, 64, 64)
    dist = dy_metrics.mlpips(pred, pred, mask, net="alex")
    assert dist < 1e-3
