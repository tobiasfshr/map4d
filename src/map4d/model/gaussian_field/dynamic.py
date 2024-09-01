from typing import Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SceneContraction
from torch import Tensor

from map4d.model.gaussian_field.base import GaussianFieldHead
from map4d.model.gaussian_field.util import scaled_sigmoid
from map4d.model.struct.gaussians import Gaussians

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class DynamicFieldHead(GaussianFieldHead):
    """Dynamic Gaussian Field Head. Used to share decoder parameters across dynamic Gaussian fields."""

    def __init__(
        self,
        feature_dim: int,
        appearance_embedding_dim: int,
        time_dependent_geometry: bool = False,
        time_dependent_appearance: bool = True,
        num_levels: int = 8,
        max_res: int = 1024,
        log2_hashmap_size: int = 16,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 64,
        implementation: str = "tcnn",
    ) -> None:
        super().__init__()
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        dir_enc_dim = self.direction_encoding.get_out_dim()

        self.feature_dim = feature_dim
        self.appearance_embedding_dim = appearance_embedding_dim
        self.time_dependent_geometry = time_dependent_geometry
        self.time_dependent_appearance = time_dependent_appearance
        self.geo_feat_dim = feature_dim

        self.time_encoding = NeRFEncoding(
            in_dim=1,
            num_frequencies=6,
            min_freq_exp=0.0,
            max_freq_exp=5.0,
            include_input=True,
            implementation=implementation,
        )
        time_enc_dim = self.time_encoding.get_out_dim()

        base_res: int = 16
        features_per_level: int = 2
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.embedding_grid = tcnn.Encoding(
            n_input_dims=4,
            encoding_config={
                "otype": "SequentialGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
                "include_static": False,
            },
        )
        self.spatial_distortion = SceneContraction()

        if time_dependent_geometry:
            self.mlp_geo = MLP(
                in_dim=self.geo_feat_dim + time_enc_dim,
                out_dim=3,
                num_layers=head_mlp_num_layers,
                layer_width=head_mlp_layer_width,
                implementation=implementation,
            )

        if not time_dependent_appearance:
            time_enc_dim = 0
        self.color_head = MLP(
            in_dim=self.geo_feat_dim + dir_enc_dim + time_enc_dim + appearance_embedding_dim,
            out_dim=3,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            implementation=implementation,
        )
        self.deformation_prior = 0.01

    def get_feature(self, gaussians: Gaussians, mask: Tensor | None):
        assert hasattr(gaussians, "object_ids")
        means, object_ids = gaussians.means, gaussians.object_ids
        if mask is not None:
            means = means[mask]
            object_ids = object_ids[mask]

        positions = self.spatial_distortion(means.detach())
        positions = (positions + 2.0) / 4.0
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        return self.embedding_grid(torch.cat([positions, object_ids], -1))

    def get_geometry(
        self, gaussians: Gaussians, mask: Tensor | None = None, time: float | None = None
    ) -> tuple[Tensor, Tensor, Tensor, dict | None]:
        means, scales, quats = gaussians.means, gaussians.scales, gaussians.quats

        if mask is not None:
            means = means[mask]
            scales = scales[mask]
            quats = quats[mask]

        feature = None
        if self.time_dependent_geometry:
            feature = self.get_feature(gaussians, mask)
            if time is None:
                time = torch.tensor([0.0]).to(means.device)
            encoded_time = self.time_encoding(time.unsqueeze(0)).repeat(means.shape[0], 1)
            geo_outs = self.mlp_geo(torch.cat([feature, encoded_time], -1)).float() * self.deformation_prior
            means = means + geo_outs

        return means, scales, quats, {"geo_feat": feature}

    def get_colors_opacities(
        self,
        gaussians: Gaussians,
        camera: Cameras,
        mask: Tensor | None = None,
        geo_outs: Optional[dict] = None,
        geometry_embedding: Tensor | None = None,
        appearance_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if "get_feat" in geo_outs and geo_outs["geo_feat"] is not None:
            feats = geo_outs["geo_feat"]
        else:
            feats = self.get_feature(gaussians, mask)

        if appearance_embedding is not None:
            appearance_embedding = appearance_embedding.unsqueeze(0).repeat(feats.shape[0], 1)
            feats = torch.cat([feats, appearance_embedding], dim=-1)

        opacities = gaussians.opacities
        if mask is not None:
            opacities = opacities[mask]

        encoded_input = self.direction_encoding(geo_outs["viewdirs"])
        if self.time_dependent_appearance:
            encoded_time = self.time_encoding(camera.times).repeat(encoded_input.shape[0], 1)
            encoded_input = torch.cat([encoded_input, encoded_time], dim=-1)

        feats = torch.cat([encoded_input, feats], dim=-1)
        return scaled_sigmoid(self.color_head(feats).float()), opacities
