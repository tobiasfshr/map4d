"""Proposal network field incl. transient embedding."""


from typing import Literal, Optional, Tuple

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor, nn


class HashMLPDensityField(Field):
    """A lightweight density field module incl transient embedding."""

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
        average_init_density: float = 1.0,
        transient_embedding_dim: int = 0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        self.average_init_density = average_init_density

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        self.transient_embedding_dim = transient_embedding_dim

        self.encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        if not self.use_linear:
            self.mlp = MLP(
                in_dim=self.encoding.get_out_dim() + transient_embedding_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
        else:
            self.linear = torch.nn.Linear(self.encoding.get_out_dim(), 1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        x = self.encoding(positions_flat).to(positions)
        if not self.use_linear:
            if self.transient_embedding_dim > 0:
                # Add transient embedding to the input of the MLP.
                embedded_transient = ray_samples.metadata["scene_transient_embeddings"].reshape(
                    -1, self.transient_embedding_dim
                )
                x = torch.cat([x, embedded_transient], dim=-1)
            density_before_activation = self.mlp(x).view(*ray_samples.frustums.shape, -1).to(positions)
        else:
            assert self.transient_embedding_dim == 0, "transient embedding not supported with linear head"
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}
