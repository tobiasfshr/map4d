"""Tests assuring that metrics are consistent with original nerfstudio metrics."""

import torch
from lpips import lpips
from pytorch_msssim import SSIM as GS_SSIM
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from map4d.common.metric import psnr, ssim

N_REPEATS = 10


def test_psnr():
    """Test PSNR metric."""
    my_psnr = psnr
    torchmetrics_psnr = PeakSignalNoiseRatio(data_range=1.0)
    for _ in range(N_REPEATS):
        x = torch.rand(1, 3, 256, 512)
        y = torch.rand(1, 3, 256, 512)
        assert torch.isclose(my_psnr(x, y), torchmetrics_psnr(x, y))


def test_ssim():
    """Test SSIM metric."""
    torchmetrics_ssim = structural_similarity_index_measure
    gs_ssim = GS_SSIM(data_range=1.0, size_average=True, channel=3)
    for _ in range(N_REPEATS):
        x = torch.rand(1, 3, 256, 512)
        y = torch.rand(1, 3, 256, 512)
        assert torch.isclose(ssim(x[0].permute(1, 2, 0), y[0].permute(1, 2, 0)), torchmetrics_ssim(x, y), atol=1e-4)
        assert torch.isclose(torchmetrics_ssim(x, y), gs_ssim(x, y), atol=1e-4)


def test_lpips():
    """Test LPIPS metric."""
    my_lpips = lpips.LPIPS(net="alex")
    torchmetrics_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
    for _ in range(N_REPEATS):
        x = torch.rand(1, 3, 256, 512)
        y = torch.rand(1, 3, 256, 512)
        assert torch.isclose(my_lpips(x, y, normalize=True), torchmetrics_lpips(x, y), atol=1e-4)
