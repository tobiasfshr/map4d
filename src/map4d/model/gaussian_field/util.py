"""Gaussian field utilities."""
import math

import numpy as np
import torch
from gsplat.sh import num_sh_bases
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from map4d.model.struct.gaussians import Gaussians

# 0th spherical harmonic coefficient
SH_C0 = 0.28209479177387814  # 1 / (4 * math.pi) ** 0.5


def scaled_sigmoid(x: Tensor, a: float = 0.9) -> Tensor:
    """Sigmoid function scaled by a factor of a."""
    return (1 / a) * torch.sigmoid(a * x)


def initialize_gaussians(
    xyz: Tensor | int,
    colors: Tensor | None = None,
    sh_degree: int = 3,
    opacity_prior: float = 0.1,
    random_scale: float = 1.0,
) -> Gaussians:
    if isinstance(xyz, int):
        xyz = (torch.rand(xyz, 3) - 0.5) * random_scale

    opacities = torch.logit(opacity_prior * torch.ones(xyz.shape[0], 1))
    scales = get_init_scales(xyz)
    quats = random_quat_tensor(xyz.shape[0])

    gaussians = Gaussians(
        means=xyz,
        opacities=opacities,
        scales=scales,
        quats=quats,
    )

    if colors is not None:
        dim_sh = num_sh_bases(sh_degree)
        shs = torch.zeros((colors.shape[0], dim_sh, 3), dtype=torch.float32, device=colors.device)
        if sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(colors / 255)
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(colors / 255, eps=1e-10)

        gaussians.add_attribute("features_dc", shs[:, 0, :].contiguous())
        gaussians.add_attribute("features_rest", shs[:, 1:, :].view(colors.shape[0], -1).contiguous())
    return gaussians


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    return (rgb - 0.5) / SH_C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    return sh * SH_C0 + 0.5


def k_nearest_sklearn(x: torch.Tensor | np.ndarray, k: int):
    """Find k-nearest neighbors using sklearn's NearestNeighbors.
    Args:
        x: The data of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
    """
    # Convert tensor to numpy array
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Build the nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def get_init_scales(means: Tensor) -> Tensor:
    """Get the initial scales for the gaussians."""
    distances, _ = k_nearest_sklearn(means, 3)
    distances = torch.from_numpy(distances)
    # find the average of the three nearest neighbors for each point and use that as the scale
    avg_dist = distances.mean(dim=-1, keepdim=True)
    scales = torch.log(avg_dist.repeat(1, 3).clamp(1e-6, 1e6))
    return scales
