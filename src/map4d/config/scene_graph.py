"""Experiment configurations."""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import LocalWriterConfig, LoggingConfig, ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from map4d.cameras.camera_optimizer import VehiclePoseOptimizerConfig
from map4d.data.dataset import DepthDataset
from map4d.data.parser.base import StreetSceneParserConfig
from map4d.data.pixel_sampler import PixelSamplerConfig
from map4d.engine.trainer import TrainerConfig
from map4d.model.scene_graph import SceneGraphModelConfig
from map4d.pipeline.pipeline import PipelineConfig

ml_nsg_vkitti2 = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="multi_level_neural_scene_graphs",
        experiment_name="vkitti2",
        method_name="ml-nsg-vkitti2",
        steps_per_eval_batch=250,
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
            ),
            model=SceneGraphModelConfig(
                appearance_embedding_dim=0,
                optimize_box_poses=False,
                transient_embedding_dim=0,
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
                camera_optimizer=VehiclePoseOptimizerConfig(),
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
            "object_embeddings": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
            },
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-08, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=100000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-3, eps=1e-15
                ),  # no weight decay since regularizer is introduced in new nerfstudio version
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Multi-level neural scene graphs VKITTI2 model.",
)


ml_nsg_kitti = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="multi_level_neural_scene_graphs",
        experiment_name="kitti",
        method_name="ml-nsg-kitti",
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
            ),
            model=SceneGraphModelConfig(
                appearance_embedding_dim=0,
                transient_embedding_dim=0,
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
                camera_optimizer=CameraOptimizerConfig(mode="SE3"),
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
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-08, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=100000),
            },
            "object_embeddings": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=100000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-5, eps=1e-8
                ),  # no weight decay since regularizer is introduced in new nerfstudio version
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=100000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Multi-level neural scene graphs KITTI model.",
)


ml_nsg_av2 = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="multi_level_neural_scene_graphs",
        experiment_name="av2_overlap",
        method_name="ml-nsg-av2",
        steps_per_eval_batch=100,
        steps_per_eval_image=1000,
        steps_per_save=2000,
        steps_per_eval_all_images=250000,
        max_num_iterations=250001,
        mixed_precision=True,
        pipeline=PipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=StreetSceneParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                train_num_images_to_sample_from=500,
                train_num_times_to_repeat_images=2500,
                eval_num_images_to_sample_from=500,
                eval_num_times_to_repeat_images=2500,
                pixel_sampler=PixelSamplerConfig(),
            ),
            model=SceneGraphModelConfig(
                eval_num_rays_per_chunk=1 << 15,
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=250000),
            },
            "dynamic_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=250000),
            },
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-08, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=250000),
            },
            "scene_embeddings": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=250000),
            },
            "object_embeddings": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=250000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-5, eps=1e-8
                ),  # no weight decay since regularizer is introduced in new nerfstudio version
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=250000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Multi-level neural scene graphs model for Argoverse 2.",
)
