"""Method configs."""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig  # noqa: F401
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from map4d.cameras.camera_optimizer import VehiclePoseOptimizerConfig
from map4d.data.datamanager import MultiProcessFullImageDatamanager, MultiProcessFullImageDatamanagerConfig
from map4d.data.dataset import DepthDataset
from map4d.data.parser.base import StreetSceneParserConfig
from map4d.data.pixel_sampler import PixelSamplerConfig
from map4d.engine.trainer import TrainerConfig
from map4d.model.gauss_splat import NeuralDynamicGaussianSplattingConfig
from map4d.model.scene_graph import SceneGraphModelConfig
from map4d.pipeline.pipeline import PipelineConfig

ml_nsg = MethodSpecification(
    config=TrainerConfig(
        project_name="multi_level_neural_scene_graphs",
        method_name="ml-nsg",
        steps_per_eval_batch=100,
        steps_per_eval_image=1000,
        steps_per_save=2000,
        steps_per_eval_all_images=100000,
        max_num_iterations=100001,
        mixed_precision=True,
        pipeline=PipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=StreetSceneParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=8192,
                pixel_sampler=PixelSamplerConfig(),
            ),
            model=SceneGraphModelConfig(
                appearance_embedding_dim=0,
                eval_num_rays_per_chunk=1 << 14,
                near_plane=0.0,
                num_nerf_samples_per_ray=64,
                num_proposal_samples_per_ray=(512, 512),
                proposal_net_args_list=[
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False},
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False},
                ],
                hidden_dim=256,
                hidden_dim_color=256,
                max_res=8192,
                proposal_weights_anneal_max_num_iters=5000,
                log2_hashmap_size=21,
                camera_optimizer=VehiclePoseOptimizerConfig(mode="SE3"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
            "static_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
            },
            "dynamic_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
            },
            "scene_embeddings": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
            },
            "object_embeddings": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
            },
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=100000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-5, eps=1e-15
                ),  # no weight decay since regularizer is introduced in new nerfstudio version
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=100000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Multi-level neural scene graphs.",
)

neural_dyn_gs = MethodSpecification(
    config=TrainerConfig(
        project_name="dynamic_gaussian_fields",
        method_name="4dgf",
        steps_per_eval_batch=10,
        steps_per_eval_image=500,
        steps_per_save=2000,
        steps_per_eval_all_images=2000,
        max_num_iterations=30001,
        mixed_precision=False,
        pipeline=PipelineConfig(
            datamanager=MultiProcessFullImageDatamanagerConfig(
                _target=MultiProcessFullImageDatamanager[DepthDataset],
                cache_images_type="uint8",
                dataparser=StreetSceneParserConfig(
                    load_pointclouds=True, remove_nonrigid=False, add_background_points=True
                ),
            ),
            model=NeuralDynamicGaussianSplattingConfig(
                # schedule params
                warmup_length=1000,
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
                # cam opt
                camera_optimizer=CameraOptimizerConfig(mode="SE3"),
                optimize_box_poses=True,
            ),
        ),
        optimizers={
            # explicit attributes
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            # neural components
            "scene_head": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-08),
                "scheduler": None,
            },
            "object_heads": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-08),
                "scheduler": None,
            },
            # codes
            "scene_embeddings": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": None,
            },
            # cameras
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-5, eps=1e-15
                ),  # no weight decay since regularizer is introduced in new nerfstudio version
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=30000),
            },
            # objects
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="4DGF model.",
)
