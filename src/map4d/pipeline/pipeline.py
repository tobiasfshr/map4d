"""Pipeline for methods using pose optimization."""

import copy
import traceback
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from map4d.common.distributed import all_gather_object_list, dict_to_cpu


@dataclass
class PipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: Pipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=lambda: DataManagerConfig())
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=lambda: DataManagerConfig())
    """specifies the model config"""
    resume: bool = False
    """specifies if the pipeline is resuming from a checkpoint"""


class Pipeline(VanillaPipeline):
    """Pipeline for methods using pose optimization.

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: PipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()  # nn.Module init
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )

        # For distributed training, force gsplat to be compiled here
        # because for multiprocessing this causes a race condition otherwise
        if world_size > 1:
            for i in range(world_size):
                if i == local_rank:
                    try:
                        from gsplat.cuda._backend import _C as _  # noqa: F401
                    except ImportError:
                        CONSOLE.log(f"Failed to import gsplat on rank {local_rank}")
                        traceback.print_exc()
                dist.barrier()

        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        with torch.cuda.device(local_rank):
            train_metadata = self._get_metadata_ddp(test_mode, local_rank, world_size, device)
            self._model = config.model.setup(
                scene_box=self.datamanager.train_dataset.scene_box,
                num_train_data=len(self.datamanager.train_dataset),
                num_eval_data=len(self.datamanager.eval_dataset),
                test_mode=test_mode,
                train_metadata=train_metadata,
                eval_metadata=self.datamanager.eval_dataset.metadata,
                device=device,
                grad_scaler=grad_scaler,
                train_cameras=self.datamanager.train_dataset.cameras,
                eval_cameras=self.datamanager.eval_dataset.cameras,
            )
            self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def _get_metadata_ddp(
        self, test_mode: Literal["test", "val", "inference"], local_rank: int, world_size: int, device: torch.device
    ):
        """Get metadata, possibly in DDP mode."""
        train_metadata = copy.deepcopy(self.datamanager.train_dataset.metadata)
        # Load point clouds if training from scratch (i.e. not test, inference mode or resume)
        if getattr(self.datamanager.dataparser.config, "load_pointclouds", False) and test_mode == "val":
            if self.config.resume:
                CONSOLE.log("WARNING: Assuming to resume from checkpoint, not loading point clouds.")
                return train_metadata

            assert hasattr(self.datamanager.dataparser, "load_points"), "load_points method not implemented"

            if local_rank == 0 or world_size <= 1:
                scene_points, scene_colors, object_points, object_colors = self.datamanager.dataparser.load_points(
                    device
                )
                results_shapes = scene_points.shape[0], [pts.shape[0] for pts in object_points]
            else:
                results_shapes = [None, None]

            if world_size > 1:
                dist.barrier()
                dist.broadcast_object_list(results_shapes)
                num_points, num_points_per_object = results_shapes
                if not local_rank == 0:
                    scene_points = torch.zeros(num_points, 3, device=device)
                    scene_colors = torch.zeros(num_points, 3, device=device)
                    object_points = [torch.zeros(num_pts, 3, device=device) for num_pts in num_points_per_object]
                    object_colors = [torch.zeros(num_pts, 3, device=device) for num_pts in num_points_per_object]

                dist.broadcast(scene_points, 0)
                dist.broadcast(scene_colors, 0)
                for i in range(len(num_points_per_object)):
                    dist.broadcast(object_points[i], 0)
                    dist.broadcast(object_colors[i], 0)

            train_metadata["seed_points"] = scene_points
            train_metadata["seed_colors"] = scene_colors
            train_metadata["object_points"] = object_points
            train_metadata["object_colors"] = object_colors
        return train_metadata

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Do not set pipeline in eval mode to enable camera optimization in DDP.

        Args:
            step: current iteration step
        """
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i, (camera, batch) in enumerate(self.datamanager.fixed_indices_eval_dataloader):
                # time this the following line
                inner_start = time()
                camera.metadata["render_images"] = False
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                dict_to_cpu(metrics_dict)

                if output_path is not None:
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path / "{0:06d}-{1}.jpg".format(i, key)
                        )
                del images_dict

                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # gather full metrics list from the processes
        metrics_dict_list = all_gather_object_list(metrics_dict_list, self.world_size)
        # average the metrics list
        metrics_dict = _average_metrics_list(metrics_dict_list, get_std)
        self.train()
        return metrics_dict


def _average_metrics_list(metrics_dict_list, get_std=True, prefix=""):
    """Average the metrics list."""
    metrics_dict = {}
    for key in metrics_dict_list[0].keys():
        metric_values = torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
        metric_values = metric_values[~torch.isnan(metric_values)]
        if get_std:
            key_std, key_mean = torch.std_mean(metric_values)
            metrics_dict[prefix + key] = float(key_mean)
            metrics_dict[prefix + f"{key}_std"] = float(key_std)
        else:
            metrics_dict[prefix + key] = float(metric_values.mean())
    return metrics_dict
