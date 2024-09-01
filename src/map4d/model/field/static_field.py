"""Static field phi. Computes the static and transient densities, as well as the rgb color."""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames, TransientDensityFieldHead, TransientRGBFieldHead
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from torch import Tensor
from torch.nn import functional as F

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class StaticField(Field):
    """Static field phi. Computes the static and transient densities, as well as the rgb color."""

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 32,
        average_init_density: float = 1.0,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim
        self.appearance_embedding_dim = appearance_embedding_dim
        self.transient_embedding_dim = transient_embedding_dim
        self.use_appearance_embedding = appearance_embedding_dim > 0
        self.use_transient_embedding = transient_embedding_dim > 0
        self.average_init_density = average_init_density

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

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            network_config=network_config,
        )

        # transients
        if self.use_transient_embedding:
            self.mlp_transient = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.transient_embedding_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_transient,
                    "n_hidden_layers": num_layers_transient - 1,
                },
            )
            self.field_head_transient_rgb = TransientRGBFieldHead(
                in_dim=self.mlp_transient.n_output_dims + self.direction_encoding.n_output_dims
            )
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.n_output_dims)

        if hidden_dim <= 128:
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            }
        else:
            network_config = {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            }

        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
            n_output_dims=3,
            network_config=network_config,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # transients
        if self.use_transient_embedding:
            embedded_transient = ray_samples.metadata["scene_transient_embeddings"]
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.reshape(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(
                torch.cat([x, d.view(*x.shape[:-1], -1)], -1)
            )
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        if self.use_appearance_embedding:
            embedded_appearance = ray_samples.metadata["scene_appearance_embeddings"]
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_appearance.reshape(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if self.use_transient_embedding:
            composite_weights = F.normalize(
                torch.cat([field_outputs[FieldHeadNames.DENSITY], field_outputs[FieldHeadNames.TRANSIENT_DENSITY]], -1),
                p=1,
                dim=-1,
                eps=1e-6,
            )
            field_outputs[FieldHeadNames.DENSITY] += field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
            field_outputs[FieldHeadNames.RGB] = (
                composite_weights[..., :1] * field_outputs[FieldHeadNames.RGB]
                + composite_weights[..., 1:] * field_outputs[FieldHeadNames.TRANSIENT_RGB]
            )

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
