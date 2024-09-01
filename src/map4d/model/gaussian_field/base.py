from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from gsplat.sh import num_sh_bases, spherical_harmonics
from nerfstudio.cameras.cameras import Cameras
from torch import Tensor, nn

from map4d.model.struct.gaussians import Gaussians


class GaussianFieldHead(nn.Module, ABC):
    """Gaussian field head."""

    def __init__(self):
        super().__init__()
        self.step = 0

    def set_step(self, step: int):
        """Set the step for the Gaussian field head."""
        self.step = step

    def get_geometry(
        self, gaussians: Gaussians, mask: Tensor | None = None, time: float | None = None
    ) -> tuple[Tensor, Tensor, Tensor, dict | None]:
        """Get geometry related attributes, i.e. mean, scale, quaternion.

        Returns also potential intermediate results that are input to the color head.

        Args:
            gaussians: 3D Gaussians
            mask: Tensor indicating to mask Gaussians for computation
            time (float): Time

        Returns:
            tuple[Tensor, Tensor, Tensor, Optional[dict]]: Mean, scale, quat, geo_outs
        """
        means, scales, quats = gaussians.means, gaussians.scales, gaussians.quats
        if mask is not None:
            return means[mask], scales[mask], quats[mask], None
        return means, scales, quats, None

    @abstractmethod
    def get_colors_opacities(
        self,
        gaussians: Gaussians,
        camera: Cameras,
        mask: Tensor | None = None,
        geo_outs: dict | None = None,
        geometry_embedding: Tensor | None = None,
        appearance_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Get color from the input features.

        Args:
            gaussians (Gaussians): Input 3D Gaussians
            camera (Cameras): Camera to compute e.g. view directions
            mask: Tensor indicating to mask Gaussians for computation
            geo_outs (dict): Intermediate outputs used in color prediction
            geometry_embedding (Tensor): geometry embedding used for opacity modulation
            appearance_embedding (Tensor): appearance embedding
            time (float): Time

        Returns:
            tuple[Tensor, Tensor]: colors, opacities.
        """
        raise NotImplementedError

    def get_relative_view_dirs(self, camera: Cameras, means: Tensor) -> Tensor:
        """Get relative view direction."""
        viewdirs = means.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        return viewdirs


class VanillaGaussianFieldHead(GaussianFieldHead):
    """Vanilla Gaussian field head."""

    def __init__(self, sh_degree: int, sh_degree_interval: int):
        super().__init__()
        self.sh_degree = sh_degree
        self.sh_degree_interval = sh_degree_interval

    def get_colors_opacities(
        self,
        gaussians: Gaussians,
        camera: Cameras,
        mask: Tensor | None = None,
        geo_outs: Optional[dict] = None,
        geometry_embedding: Tensor | None = None,
        appearance_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply spherical harmonics to the input features and return the resulting RGB values."""
        if self.sh_degree == 0:
            assert "features_dc" in gaussians._other_keys
            feats = gaussians.features_dc
        else:
            assert "features_dc" in gaussians._other_keys and "features_rest" in gaussians._other_keys
            feats = torch.cat([gaussians.features_dc, gaussians.features_rest], dim=-1).view(
                -1, num_sh_bases(self.sh_degree), 3
            )

        opacities = gaussians.opacities
        means = gaussians.means
        if mask is not None:
            means = means[mask]
            opacities = opacities[mask]
            feats = feats[mask]

        if self.sh_degree > 0:
            n = min(self.step // self.sh_degree_interval, self.sh_degree)
            viewdirs = self.get_relative_view_dirs(camera, means)
            rgbs = spherical_harmonics(n, viewdirs, feats)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(feats)
        return rgbs, opacities
