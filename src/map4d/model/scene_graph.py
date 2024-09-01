"""Multi-level neural scene graph. We combine the static and dynamic fields during rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from lpips import lpips
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.lie_groups import exp_map_SE3
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import UniformLinDispPiecewiseSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions
from torch.nn import Parameter
from torch.nn import functional as F

from map4d.cameras.camera_optimizer import VehiclePoseOptimizerConfig
from map4d.common.metric import depth_metrics, psnr, ssim
from map4d.common.visualize import apply_depth_colormap
from map4d.cuda import VideoEmbedding, ray_box_intersect
from map4d.model.field.density_field import HashMLPDensityField
from map4d.model.field.dynamic_field import DynamicField
from map4d.model.field.static_field import StaticField
from map4d.model.loss.entropy import compute_entropy
from map4d.model.sampler.composite_proposal_sampler import CompositeProposalNetworkSampler
from map4d.model.util import calculate_local_ray_samples, get_objects_per_ray


@dataclass
class SceneGraphModelConfig(ModelConfig):
    """Multi-level neural scene graph model config"""

    _target: Type = field(default_factory=lambda: SceneGraphModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "black"
    """Whether to randomize the background color."""
    average_init_density: float = 0.01
    """Average initial density for the static field."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    appearance_embedding_dim: int = 32
    """Whether to use appearance embedding"""
    transient_embedding_dim: int = 32
    """Whether to use transient embedding"""
    scene_embedding_num_frequencies: int = 6
    """Number of frequencies in time encoding for scene embeddings."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Whether to scale gradients by distance squared."""

    # object settings
    optimize_box_poses: bool = True
    """Whether to optimize box poses or not."""
    object_embedding_dim: int = 32
    """Dimension of the object embeddings."""
    semantic_prior_load_dir: str = "assets/semantic_prior.ckpt"
    """Path to the semantic prior model."""

    # depth loss params
    depth_loss_mult: float = 5e-2
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""

    # depth visualization settings
    min_depth: float = 1.0
    """Minimum depth for visualization."""
    max_depth: float = 82.5
    """Maximum depth for visualization."""

    # entropy loss params
    entropy_loss_mult: float = 1e-4
    """Entropy loss coefficient"""
    entropy_skewness: float = 1.0
    """Skewedness of the entropy."""

    # camera optimizer params
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: VehiclePoseOptimizerConfig(mode="SE3"))
    """Config of the camera optimizer to use."""


class SceneGraphModel(Model):
    """Multi-level neural scene graph model

    Args:
        config: configuration to instantiate model
    """

    config: SceneGraphModelConfig

    def __init__(self, *args, num_eval_data: int, test_mode: str = "val", **kwargs):
        train_meta = kwargs.get("train_metadata", None)
        eval_meta = kwargs.get("eval_metadata", None)
        assert train_meta is not None and eval_meta is not None, "Missing metadata"
        self.num_eval_data = num_eval_data
        self.num_cameras = train_meta["num_cameras"]
        self.dataparser_scale = train_meta["dataparser_scale"]
        self.test_mode = test_mode  # if 'inference' we do not want to apply pose adjustments to the ray bundles

        # object info
        self.object_seq_ids = train_meta["object_seq_ids"]
        self.object_times = train_meta["object_times"]
        self.object_poses = train_meta["object_poses"]
        self.object_dims = train_meta["object_dims"]
        self.object_ids = train_meta["object_ids"]

        self.num_objects = (
            torch.cat(self.object_ids).unique().numel()
            if isinstance(self.object_ids, list)
            else self.object_ids.unique().numel()
        )
        assert (torch.cat(self.object_ids).unique() == torch.arange(self.num_objects)).all()

        # transform lists to tensors while preserving dimensions, pad entries
        if isinstance(self.object_ids, list):

            def _to_tensor(x, pad_val=0):
                max_shape = torch.tensor([y.shape for y in x]).max(dim=0)[0]
                x_tensor = torch.ones((len(x), *max_shape)) * pad_val
                for i, y in enumerate(x):
                    x_tensor[i, : len(y)] = y
                return x_tensor

            self.object_ids = _to_tensor(self.object_ids, pad_val=-1)
            self.object_poses = _to_tensor(self.object_poses)
            self.object_dims = _to_tensor(self.object_dims)

        # scene info
        self.sequence_ids = train_meta["sequence_ids"].unique()
        assert (self.sequence_ids == torch.arange(self.sequence_ids.numel())).all()

        super().__init__(*args, **kwargs)

        # object related info
        self.object_poses = Parameter(self.object_poses, requires_grad=False)
        self.object_poses_delta = Parameter(
            torch.zeros_like(self.object_poses), requires_grad=self.config.optimize_box_poses
        )
        self.object_dims = Parameter(self.object_dims, requires_grad=False)
        self.object_ids = Parameter(self.object_ids, requires_grad=False)
        self.object_seq_ids = Parameter(self.object_seq_ids, requires_grad=False)
        self.object_times = Parameter(self.object_times, requires_grad=False)

    def populate_modules(self):
        """Set the fields and modules."""
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Node latent codes
        # scene level codes
        num_seq = self.sequence_ids.numel()
        if num_seq > 1 and self.config.appearance_embedding_dim > 0:
            # Dimension of fourier encoding applied to time when computing appearance embeddings
            self.scene_embedding_appearance = VideoEmbedding(
                num_seq, self.config.scene_embedding_num_frequencies, self.config.appearance_embedding_dim
            )

        if self.config.transient_embedding_dim > 0:
            self.scene_embedding_transient = VideoEmbedding(
                num_seq, self.config.scene_embedding_num_frequencies, self.config.transient_embedding_dim
            )

        # object level codes
        self.object_shape_embeddings = Embedding(self.num_objects, self.config.object_embedding_dim)
        self.object_appearance_embeddings = Embedding(self.num_objects, self.config.object_embedding_dim)

        # Fields
        # static field
        self.static_field = StaticField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim if num_seq > 1 else 0,
            transient_embedding_dim=self.config.transient_embedding_dim,
            average_init_density=self.config.average_init_density,
        )
        # dynamic field
        self.dynamic_field = DynamicField(
            object_embedding_dim=self.config.object_embedding_dim,
            scene_embedding_dim=self.config.appearance_embedding_dim if num_seq > 1 else 0,
            load_dir=self.config.semantic_prior_load_dir,
        )

        self.static_density_fns = []
        self.dynamic_density_fns = [self.dynamic_density_fn for _ in range(self.config.num_proposal_iterations)]
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            static_network = HashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args
            )
            self.proposal_networks.append(static_network)
            self.static_density_fns.extend([lambda x: static_network.get_density(x)[0] for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                static_network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    transient_embedding_dim=self.config.transient_embedding_dim,
                    average_init_density=self.config.average_init_density,
                    **prop_net_args,
                )
                self.proposal_networks.append(static_network)
                self.static_density_fns.append(lambda x: static_network.get_density(x)[0])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        if self.config.num_proposal_iterations > 0:
            self.proposal_sampler = CompositeProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
                num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
                num_proposal_network_iterations=self.config.num_proposal_iterations,
                single_jitter=self.config.use_single_jitter,
                update_sched=update_schedule,
                initial_sampler=initial_sampler,
            )
        else:
            self.uniform_sampler = UniformLinDispPiecewiseSampler(
                num_samples=self.config.num_nerf_samples_per_ray, single_jitter=self.config.use_single_jitter
            )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.depth_loss = MSELoss()
        self.rgb_loss = MSELoss()

        # camera optimizer
        camopt_kwargs = dict(device="cpu")
        if isinstance(self.config.camera_optimizer, VehiclePoseOptimizerConfig):
            camopt_kwargs["num_physical_cameras"] = self.num_cameras

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data + self.num_eval_data, **camopt_kwargs
        )

        # metrics
        self.lpips = lpips.LPIPS(net="alex")

        self.ray_bundle_results = None

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # field params
        if self.config.num_proposal_iterations > 0:
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["static_field"] = list(self.static_field.parameters())
        param_groups["dynamic_field"] = list(self.dynamic_field.parameters())

        # scene level codes
        param_groups["scene_embeddings"] = []
        if self.sequence_ids.numel() > 1 and self.config.appearance_embedding_dim > 0:
            param_groups["scene_embeddings"] += list(self.scene_embedding_appearance.parameters())
        if self.config.transient_embedding_dim > 0:
            param_groups["scene_embeddings"] += list(self.scene_embedding_transient.parameters())
        if not len(param_groups["scene_embeddings"]):
            del param_groups["scene_embeddings"]

        # object level codes
        param_groups["object_embeddings"] = list(self.object_shape_embeddings.parameters()) + list(
            self.object_appearance_embeddings.parameters()
        )

        # camera and object poses
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        if self.config.optimize_box_poses:
            param_groups["object_poses"] = [self.object_poses_delta]
        return param_groups

    def precompute_box_intersection_for_ray_bundle(self, ray_bundle: RayBundle):
        # apply delta for box param optim
        object_poses = self.object_poses.clone()
        object_poses = object_poses + self.object_poses_delta

        # add camera opt params with no grad. Can only be done if camera optimizer is VehiclePoseOptimizer
        if self.config.camera_optimizer.mode != "off" and isinstance(
            self.config.camera_optimizer, VehiclePoseOptimizerConfig
        ):
            all_pose_adjustment = self.camera_optimizer.pose_adjustment.detach()
            vehicle_poses = exp_map_SE3(all_pose_adjustment)
            object_poses[..., :3] = (vehicle_poses[:, None, :3, :3] @ object_poses[..., :3, None]).squeeze(
                -1
            ) + vehicle_poses[:, None, :3, 3]
            object_poses[..., 3] += torch.arctan2(vehicle_poses[..., 1, 0], vehicle_poses[..., 0, 0]).unsqueeze(-1)

        if "sequence_ids" not in ray_bundle.metadata:
            # when rendering a nerfstudio viewer trajectory, just take the first sequence
            seq_ids = torch.ones_like(ray_bundle.times) * self.object_seq_ids[0]
        else:
            seq_ids = ray_bundle.metadata["sequence_ids"]

        per_ray_boxes3d, per_ray_obj_ids = get_objects_per_ray(
            seq_ids,
            ray_bundle.times,
            self.object_seq_ids,
            self.object_times,
            self.object_ids,
            self.object_dims,
            object_poses,
        )

        # get point-box intersections and local coordinates / directions
        with torch.no_grad():
            local_origins, local_directions, near_fars, hit_mask = ray_box_intersect(
                ray_bundle.origins.float(), ray_bundle.directions.float(), per_ray_boxes3d.float()
            )

        # transform rays into local space w/ grad if box poses are optimized
        if self.config.optimize_box_poses:
            local_origins = ray_bundle.origins.unsqueeze(1) - per_ray_boxes3d[..., :3]
            yaw = -per_ray_boxes3d[..., 6]
            cos, sin, zeros, ones = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
            rotation = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], -1).view(*yaw.shape, 3, 3)
            local_origins = torch.matmul(rotation, local_origins.unsqueeze(-1)).squeeze(-1)
            local_directions = torch.matmul(rotation, ray_bundle.directions[:, None, :, None]).squeeze(-1)

        if hit_mask.any():
            self.ray_bundle_results = [
                per_ray_boxes3d,
                per_ray_obj_ids,
                local_origins,
                local_directions,
                near_fars,
                hit_mask,
            ]
        else:
            self.ray_bundle_results = None

    def dynamic_density_fn(self, ray_samples: RaySamples):
        # get cached results
        if self.ray_bundle_results is None:
            return torch.zeros_like(ray_samples.frustums.origins[..., 0:1])

        per_ray_boxes3d, per_ray_obj_ids, local_origins, local_directions, near_fars, _ = self.ray_bundle_results

        obj_ids, _, local_origins, local_directions, starts, ends, hit_mask = calculate_local_ray_samples(
            ray_samples.frustums.starts,
            ray_samples.frustums.ends,
            per_ray_boxes3d,
            per_ray_obj_ids,
            local_origins,
            local_directions,
            near_fars,
        )

        density = torch.zeros_like(ray_samples.frustums.origins[..., 0:1])
        if not hit_mask.any():
            return density

        assert (obj_ids >= 0).all()
        shape_embeds = self.object_shape_embeddings(obj_ids)

        local_ray_samples = RaySamples(
            frustums=Frustums(
                local_origins, local_directions, starts, ends, pixel_area=ray_samples.frustums.pixel_area[hit_mask]
            ),
            metadata={"object_shape_embeddings": shape_embeds},
        )
        density[hit_mask], _ = self.dynamic_field.get_density(local_ray_samples)
        return density

    def get_dynamic_outputs(self, ray_samples: RaySamples):
        if self.ray_bundle_results is None:
            field_outputs = {}
            field_outputs[FieldHeadNames.DENSITY] = torch.zeros_like(ray_samples.frustums.origins[..., 0:1])
            field_outputs[FieldHeadNames.RGB] = torch.zeros_like(ray_samples.frustums.origins)
            field_outputs["hit_mask"] = torch.zeros_like(ray_samples.frustums.origins[..., 0], dtype=torch.bool)
            return field_outputs

        per_ray_boxes3d, per_ray_obj_ids, local_origins, local_directions, near_fars, _ = self.ray_bundle_results

        obj_ids, _, local_origins, local_directions, starts, ends, hit_mask = calculate_local_ray_samples(
            ray_samples.frustums.starts,
            ray_samples.frustums.ends,
            per_ray_boxes3d,
            per_ray_obj_ids,
            local_origins,
            local_directions,
            near_fars,
        )

        if not hit_mask.any():
            field_outputs = {}
            field_outputs[FieldHeadNames.DENSITY] = torch.zeros_like(ray_samples.frustums.origins[..., 0:1])
            field_outputs[FieldHeadNames.RGB] = torch.zeros_like(ray_samples.frustums.origins)
            field_outputs["hit_mask"] = torch.zeros_like(ray_samples.frustums.origins[..., 0], dtype=torch.bool)
            return field_outputs

        # get field inputs
        shape_embeds = self.object_shape_embeddings(obj_ids)
        app_embeds = self.object_appearance_embeddings(obj_ids)
        ray_metadata = {"object_shape_embeddings": shape_embeds, "object_appearance_embeddings": app_embeds}
        if self.sequence_ids.numel() > 1 and self.config.appearance_embedding_dim > 0:
            ray_metadata["scene_appearance_embeddings"] = ray_samples.metadata["scene_appearance_embeddings"][hit_mask]

        local_ray_samples = RaySamples(
            frustums=Frustums(
                local_origins, local_directions, starts, ends, pixel_area=ray_samples.frustums.pixel_area[hit_mask]
            ),
            metadata=ray_metadata,
        )
        # get field outputs
        field_outputs = self.dynamic_field(local_ray_samples)

        # fill outputs with zeros where there is no intersection
        density = torch.zeros_like(ray_samples.frustums.origins[..., 0:1])
        density[hit_mask] = field_outputs[FieldHeadNames.DENSITY]
        field_outputs[FieldHeadNames.DENSITY] = density

        rgb = torch.zeros_like(ray_samples.frustums.origins, dtype=field_outputs[FieldHeadNames.RGB].dtype)
        rgb[hit_mask] = field_outputs[FieldHeadNames.RGB]
        field_outputs[FieldHeadNames.RGB] = rgb

        field_outputs["hit_mask"] = hit_mask
        return field_outputs

    def get_static_outputs(self, ray_samples: RaySamples):
        field_outputs = self.static_field(ray_samples)
        if self.config.transient_embedding_dim > 0:
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
        return field_outputs

    def _composite_outputs(self, field_outputs, dynamic_outputs, hit_mask):
        assert not torch.isnan(dynamic_outputs[FieldHeadNames.DENSITY]).any()
        assert not torch.isnan(field_outputs[FieldHeadNames.DENSITY]).any()

        static_density, dynamic_density = field_outputs[FieldHeadNames.DENSITY], dynamic_outputs[FieldHeadNames.DENSITY]

        composite_weights = torch.cat([torch.ones_like(static_density), torch.zeros_like(dynamic_density)], -1)
        composite_weights[hit_mask] = F.normalize(
            torch.cat([static_density[hit_mask], dynamic_density[hit_mask]], -1), p=1, dim=-1, eps=1e-6
        )

        field_outputs[FieldHeadNames.DENSITY] += dynamic_density
        field_outputs[FieldHeadNames.RGB] = (
            composite_weights[..., :1] * field_outputs[FieldHeadNames.RGB]
            + composite_weights[..., 1:] * dynamic_outputs[FieldHeadNames.RGB]
        )

        return field_outputs, composite_weights

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        # map camera indices (local train / eval split index) to global indices
        # since we have a single camera optimizer for both train and eval views
        ray_bundle.camera_indices = ray_bundle.metadata["cam_idx"]
        # apply camera pose deltas
        if self.training or self.test_mode != "inference":
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        # precompute box intersection for ray bundle
        self.precompute_box_intersection_for_ray_bundle(ray_bundle)

        # add seq ids for viewer
        if "sequence_ids" not in ray_bundle.metadata:
            ray_bundle.metadata["sequence_ids"] = torch.zeros_like(ray_bundle.times, dtype=torch.long)

        # add latent codes to ray samples
        if self.config.appearance_embedding_dim > 0:
            ray_bundle.metadata["scene_appearance_embeddings"] = self.scene_embedding_appearance(
                ray_bundle.times, ray_bundle.metadata["sequence_ids"]
            )
        if self.config.transient_embedding_dim > 0:
            ray_bundle.metadata["scene_transient_embeddings"] = self.scene_embedding_transient(
                ray_bundle.times, ray_bundle.metadata["sequence_ids"]
            )

        # proposal sampling
        if self.config.num_proposal_iterations > 0:
            (
                ray_samples,
                weights_list,
                ray_samples_list,
                static_weights_list,
                dynamic_weights_list,
            ) = self.proposal_sampler(ray_bundle, self.static_density_fns, self.dynamic_density_fns)
        else:
            ray_samples = self.uniform_sampler(ray_bundle)
            weights_list = []
            ray_samples_list = []
            static_weights_list = []
            dynamic_weights_list = []

        # static field
        field_outputs = self.get_static_outputs(ray_samples)

        # dynamic field
        dynamic_outputs = self.get_dynamic_outputs(ray_samples)
        assert "hit_mask" in dynamic_outputs
        hit_mask = dynamic_outputs.pop("hit_mask")

        static_weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        rgb_static = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=static_weights)
        static_weights_list.append(static_weights)

        dynamic_weights = ray_samples.get_weights(dynamic_outputs[FieldHeadNames.DENSITY])
        rgb_dynamic = self.renderer_rgb(rgb=dynamic_outputs[FieldHeadNames.RGB], weights=dynamic_weights)
        dynamic_weights_list.append(dynamic_weights)

        field_outputs, composite_weights = self._composite_outputs(field_outputs, dynamic_outputs, hit_mask)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        # rendering
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # compute entropy
        entropy = compute_entropy(composite_weights, self.config.entropy_skewness)

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        outputs = {
            "rgb": rgb,
            "rgb_static": rgb_static,
            "rgb_dynamic": rgb_dynamic,
            "accumulation": accumulation,
            "entropy": entropy,
            "depth": depth,
            "hit_mask": hit_mask.any(dim=-1),
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["static_weights_list"] = static_weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = psnr(outputs["rgb"], image)
        if self.training:
            # distortion loss only on static weights
            metrics_dict["distortion"] = distortion_loss(outputs["static_weights_list"], outputs["ray_samples_list"])

            # log object refinement magnitude
            if self.config.optimize_box_poses:
                metrics_dict["object_pose_delta"] = self.object_poses_delta.abs().mean()

            if self.config.depth_loss_mult > 0:
                metrics_dict["depth_loss"] = 0.0
                termination_depth = batch["depth_image"].to(self.device).float()

                depth_gt = termination_depth[termination_depth > 0]
                depth_keys = [f"prop_depth_{i}" for i in range(self.config.num_proposal_iterations)] + ["depth"]

                for i in range(len(depth_keys)):
                    pred_depth = outputs[depth_keys[i]]
                    if not self.config.is_euclidean_depth:
                        pred_depth *= outputs["directions_norm"]
                    metrics_dict["depth_loss"] += self.depth_loss(
                        pred_depth[termination_depth > 0].view(depth_gt.shape), depth_gt
                    ) / len(depth_keys)

            # static / dynamic entropy loss
            if self.config.entropy_loss_mult > 0:
                metrics_dict["entropy_loss"] = outputs["entropy"][outputs["hit_mask"]].mean()

            if self.sequence_ids.numel() > 1 and self.config.appearance_embedding_dim > 0:
                app_std, app_mean = torch.std_mean(self.scene_embedding_appearance.sequence_code_weights)
                metrics_dict["scene_embedding_appearance_mean"] = app_mean
                metrics_dict["scene_embedding_appearance_stddev"] = app_std

            if self.config.transient_embedding_dim > 0:
                trans_std, trans_mean = torch.std_mean(self.scene_embedding_transient.sequence_code_weights)
                metrics_dict["scene_embedding_transient_mean"] = trans_mean
                metrics_dict["scene_embedding_transient_stddev"] = trans_std

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["main_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.training:
            # interlevel loss only on static weights
            if self.config.num_proposal_iterations > 0:
                static_interlevel_loss = interlevel_loss(outputs["static_weights_list"], outputs["ray_samples_list"])
                loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * static_interlevel_loss
                assert metrics_dict is not None and "distortion" in metrics_dict
                loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

            # depth loss
            if self.config.depth_loss_mult > 0:
                assert metrics_dict is not None and "depth_loss" in metrics_dict
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]

            # static / dynamic entropy loss
            if self.config.entropy_loss_mult > 0:
                loss_dict["entropy_loss"] = self.config.entropy_loss_mult * outputs["entropy"].mean()

        # Add loss from camera optimizer
        self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Get image metrics and images for logging."""
        image = batch["image"].clone().to(self.device)
        rgb = outputs["rgb"].clone().detach()

        acc = colormaps.apply_colormap(outputs["accumulation"].detach())
        depth = apply_depth_colormap(
            outputs["depth"].detach(),
            self.config.min_depth * self.dataparser_scale,
            self.config.max_depth * self.dataparser_scale,
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # mask images for metrics computation
        if "mask" in batch:
            mask = batch["mask"].squeeze(-1)
            image[mask == False] = 0  # noqa: E712
            rgb[mask == False] = 0  # noqa: E712

        # ssim needs to be HWC
        ssim_val = ssim(image, rgb)
        psnr_val = psnr(image, rgb)

        # Switch images from [H, W, C] to [1, C, H, W] for lpips computation
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...].clamp(0.0, 1.0)
        lpips = self.lpips(image, rgb, normalize=True)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr_val.item()), "ssim": float(ssim_val)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # depth metrics (original scale)
        metrics_dict.update(
            depth_metrics(
                outputs["depth"] / self.dataparser_scale, batch["depth_image"].float() / self.dataparser_scale
            )
        )

        images_dict = {
            "img": torch.cat([combined_rgb, outputs["rgb_static"], outputs["rgb_dynamic"]], dim=1).detach(),
            "accumulation": acc,
            "depth": depth,
        }

        psnr_perpixel = (
            psnr(batch["image"].clone().to(self.device), outputs["rgb"].clone().detach(), reduction="none")
            .mean(dim=-1, keepdim=True)
            .clamp(0.0, 100.0)
        )
        psnr_perpixel = (psnr_perpixel - psnr_perpixel.min()) / (psnr_perpixel.max() - psnr_perpixel.min())
        images_dict["psnr_heatmap"] = colormaps.apply_colormap(psnr_perpixel)

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = apply_depth_colormap(
                outputs[key],
                self.config.min_depth * self.dataparser_scale,
                self.config.max_depth * self.dataparser_scale,
            )
            images_dict[key] = prop_depth_i

        images_dict["entropy"] = colormaps.apply_colormap(
            outputs["entropy"].detach(), colormap_options=ColormapOptions("inferno")
        )
        return metrics_dict, images_dict
