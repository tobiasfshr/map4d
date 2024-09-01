"""Dynamic field psi. It represents dynamic objects in the scene."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field
from torch import Tensor, nn


class DynamicField(Field):
    def __init__(
        self,
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[FieldHead]] = (RGBFieldHead(),),
        object_embedding_dim: int = 32,
        scene_embedding_dim: int = 32,
        load_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.object_embedding_dim = object_embedding_dim

        # fields
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim() + object_embedding_dim,
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim() + object_embedding_dim,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

        if load_dir is not None:
            model_state_dict = torch.load(load_dir, map_location="cpu")
            self.load_state_dict(model_state_dict)

        if scene_embedding_dim > 0:
            self.fuse_scene_appearance = nn.Linear(object_embedding_dim + scene_embedding_dim, object_embedding_dim)
        else:
            self.fuse_scene_appearance = None

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        assert "object_shape_embeddings" in ray_samples.metadata, "Missing object shape embeddings."
        shape_embeds = ray_samples.metadata["object_shape_embeddings"]
        encoded_xyz = self.position_encoding(ray_samples.frustums.get_positions())
        base_mlp_out = self.mlp_base(torch.cat([encoded_xyz, shape_embeds], -1))
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        app_embeds = ray_samples.metadata["object_appearance_embeddings"]
        if self.fuse_scene_appearance is not None:
            assert "scene_appearance_embeddings" in ray_samples.metadata, "Missing scene appearance embeddings."
            app_embeds = self.fuse_scene_appearance(
                torch.cat([app_embeds, ray_samples.metadata["scene_appearance_embeddings"]], -1)
            )

        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding, app_embeds], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
