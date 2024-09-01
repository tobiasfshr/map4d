from typing import Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from torch import Tensor
from torch.nn import functional as F

from map4d.model.gaussian_field.base import GaussianFieldHead
from map4d.model.gaussian_field.util import scaled_sigmoid
from map4d.model.struct.gaussians import Gaussians

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class StaticFieldHead(GaussianFieldHead):
    """Static Gaussian Field Head."""

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 32,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ):
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.appearance_embedding_dim = appearance_embedding_dim
        self.transient_embedding_dim = transient_embedding_dim
        self.use_appearance_embedding = appearance_embedding_dim > 0
        self.use_transient_embedding = transient_embedding_dim > 0

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        self.spatial_distortion = spatial_distortion

        base_res: int = 16
        features_per_level: int = 2
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.features_per_level = features_per_level
        self.base_res = base_res
        self.growth_factor = growth_factor

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        if hidden_dim <= 128:
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            }
        else:
            network_config = {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            }

        self.base_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
        )
        self.geo_feat_dim = self.base_grid.n_output_dims

        # transients
        if self.use_transient_embedding:
            self.mlp_transient = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.transient_embedding_dim + 1,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_transient,
                    "n_hidden_layers": num_layers_transient,
                },
            )

        if hidden_dim <= 128:
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            }
        else:
            network_config = {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            }

        self.mlp_color = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
            n_output_dims=3,
            network_config=network_config,
        )
        self.deformation_prior = 0.01

    def get_feature(self, gaussians: Gaussians, mask: Tensor | None = None):
        means = gaussians.means
        if mask is not None:
            means = means[mask]

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(means.detach())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(means.detach(), self.aabb)

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        return self.base_grid(positions)

    def get_colors_opacities(
        self,
        gaussians: Gaussians,
        camera: Cameras,
        mask: Tensor | None = None,
        geo_outs: Optional[dict] = None,
        geometry_embedding: Tensor | None = None,
        appearance_embedding: Tensor | None = None,
    ) -> Tensor:
        feature = self.get_feature(gaussians, mask)

        means = gaussians.means
        opacities = gaussians.opacities
        if mask is not None:
            means = means[mask]
            opacities = opacities[mask]

        # opacity modulation
        if geometry_embedding is not None:
            assert self.use_transient_embedding
            if len(geometry_embedding.shape) == 1:
                geometry_embedding = geometry_embedding.unsqueeze(0).repeat(feature.shape[0], 1)
            else:
                assert geometry_embedding.shape[0] == feature.shape[0]
            cond_opacities = self.mlp_transient(
                torch.cat([opacities.detach(), feature, geometry_embedding], dim=-1)
            ).float()
            # needs to multiply opacities and convert back to logit
            # more stable: log space addition, exponentiate, convert to logit
            opacities = (-F.softplus(-opacities) + -F.softplus(-cond_opacities)).exp()
            opacities = torch.logit(opacities, eps=1e-6)

        encoded_dirs = self.direction_encoding(self.get_relative_view_dirs(camera, means))
        feature = torch.cat([encoded_dirs, feature], dim=-1)

        # appearance modeling
        if appearance_embedding is not None:
            assert self.use_appearance_embedding
            if len(appearance_embedding.shape) == 1:
                appearance_embedding = appearance_embedding.unsqueeze(0).repeat(feature.shape[0], 1)
            else:
                assert appearance_embedding.shape[0] == feature.shape[0]
            feature = torch.cat([feature, appearance_embedding], dim=-1)

        return scaled_sigmoid(self.mlp_color(feature).float()), opacities
