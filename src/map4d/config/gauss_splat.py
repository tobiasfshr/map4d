from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig  # noqa: F401
from nerfstudio.configs.base_config import LocalWriterConfig, LoggingConfig, ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from map4d.data.datamanager import MultiProcessFullImageDatamanager, MultiProcessFullImageDatamanagerConfig
from map4d.data.dataset import DepthDataset
from map4d.data.parser.base import StreetSceneParserConfig
from map4d.engine.trainer import TrainerConfig
from map4d.model.gauss_splat import NeuralDynamicGaussianSplattingConfig
from map4d.pipeline.pipeline import PipelineConfig

neural_dyn_gs_vkitti = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="4dgf_vkitti_benchmark",
        method_name="4dgf-vkitti",
        steps_per_eval_batch=100,
        steps_per_eval_image=500,
        steps_per_save=2000,
        steps_per_eval_all_images=2000,
        max_num_iterations=30001,
        mixed_precision=False,
        pipeline=PipelineConfig(
            datamanager=MultiProcessFullImageDatamanagerConfig(
                cache_images_type="uint8",
                dataparser=StreetSceneParserConfig(
                    load_pointclouds=True, remove_nonrigid=False, add_background_points=False
                ),
            ),
            model=NeuralDynamicGaussianSplattingConfig(
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
            ),
        ),
        optimizers={
            # explicit attributes
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
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
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
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
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="4DGF model for VKITTI2.",
)


neural_dyn_gs_kitti = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="4dgf_kitti_benchmark",
        method_name="4dgf-kitti",
        steps_per_eval_batch=10,
        steps_per_eval_image=500,
        steps_per_save=2000,
        steps_per_eval_all_images=2000,
        max_num_iterations=30001,
        mixed_precision=False,
        pipeline=PipelineConfig(
            datamanager=MultiProcessFullImageDatamanagerConfig(
                cache_images_type="uint8",
                dataparser=StreetSceneParserConfig(
                    load_pointclouds=True, remove_nonrigid=False, add_background_points=False
                ),
            ),
            model=NeuralDynamicGaussianSplattingConfig(
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
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
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
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
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
        vis="wandb",
    ),
    description="4DGF model for KITTI.",
)


MAX_STEPS = 100_000
neural_dyn_gs_av2 = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="4dgf_av2_benchmark",
        method_name="4dgf-av2",
        steps_per_eval_batch=20,
        steps_per_eval_image=MAX_STEPS // 200,
        steps_per_save=MAX_STEPS // 20,
        steps_per_eval_all_images=MAX_STEPS // 20,
        max_num_iterations=MAX_STEPS + 1,
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
                depth_loss_mult=0.05,
                # refine
                refine_every=int(0.005 * MAX_STEPS),
                warmup_length=int(0.015 * MAX_STEPS),
                stop_split_at=int(0.6 * MAX_STEPS),
                stop_screen_size_at=int(0.2 * MAX_STEPS),
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
                    max_steps=MAX_STEPS,
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00025, max_steps=MAX_STEPS),
            },
            "object_heads": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00025, max_steps=MAX_STEPS),
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=MAX_STEPS),
            },
            # objects
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=MAX_STEPS),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="4DGF model for AV2.",
)


MAX_STEPS = 1_000_000
neural_dyn_gs_av2_big = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="4dgf_av2_benchmark",
        method_name="4dgf-av2-big",
        steps_per_eval_batch=20,
        steps_per_eval_image=MAX_STEPS // 200,
        steps_per_save=MAX_STEPS // 20,
        steps_per_eval_all_images=MAX_STEPS // 20,
        max_num_iterations=MAX_STEPS + 1,
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
                depth_loss_mult=0.05,
                # refine
                refine_every=int(0.005 * MAX_STEPS),
                warmup_length=int(0.015 * MAX_STEPS),
                stop_split_at=int(0.6 * MAX_STEPS),
                stop_screen_size_at=int(0.2 * MAX_STEPS),
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
                    max_steps=MAX_STEPS,
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00025, max_steps=MAX_STEPS),
            },
            "object_heads": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00025, max_steps=MAX_STEPS),
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=MAX_STEPS),
            },
            # objects
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=MAX_STEPS),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="4DGF model for AV2.",
)


MAX_STEPS = 60_000
neural_dyn_gs_waymo = MethodSpecification(
    config=TrainerConfig(
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=True, max_log_size=0)),
        project_name="4dgf_waymo_benchmark",
        method_name="4dgf-waymo",
        steps_per_eval_batch=10,
        steps_per_eval_image=MAX_STEPS // 60,
        steps_per_save=MAX_STEPS // 20,
        steps_per_eval_all_images=MAX_STEPS // 20,
        max_num_iterations=MAX_STEPS + 1,
        mixed_precision=False,
        pipeline=PipelineConfig(
            datamanager=MultiProcessFullImageDatamanagerConfig(
                _target=MultiProcessFullImageDatamanager[DepthDataset],
                cache_images_type="uint8",
                dataparser=StreetSceneParserConfig(
                    load_pointclouds=True, remove_nonrigid=False, add_background_points=True, orient_center_poses=False
                ),
            ),
            model=NeuralDynamicGaussianSplattingConfig(
                depth_loss_mult=0.05,
                # indicates dataset masks are dynamic object masks, s.t. it will evaluate separately
                is_dynamic_mask=True,
                # refine
                refine_every=int(0.005 * MAX_STEPS),
                warmup_length=int(0.015 * MAX_STEPS),
                stop_split_at=int(0.6 * MAX_STEPS),
                stop_screen_size_at=int(0.2 * MAX_STEPS),
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
                    max_steps=MAX_STEPS,
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00025, max_steps=MAX_STEPS),
            },
            "object_heads": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00025, max_steps=MAX_STEPS),
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
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=MAX_STEPS),
            },
            # objects
            "object_poses": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=MAX_STEPS),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="4DGF model for Waymo.",
)
