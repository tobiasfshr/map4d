"""Trainer for methods using pose optimization."""

import functools
from dataclasses import dataclass, field
from typing import Literal, Type, cast

import torch
from nerfstudio.engine.trainer import TRAIN_INTERATION_OUTPUT, Trainer, TrainerConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter

from map4d.engine.util import scale_by_world_size


@dataclass
class TrainerConfig(TrainerConfig):
    """Configuration for pose optimization trainer instantiation"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    scale_lrs: Literal["none", "linear", "sqrt"] = "none"
    """specifies the learning rate scaling method for multi gpu."""


class Trainer(Trainer):
    """Trainer with pose optimization during eval batches."""

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        scale_by_world_size(config, world_size)
        torch.cuda.set_device(local_rank)
        config.pipeline.resume = config.load_dir is not None or config.load_checkpoint is not None
        super().__init__(config, local_rank, world_size)

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())

        if not torch.isfinite(loss):
            raise ValueError(f"Train Loss is not finite: {loss_dict}.")

        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            self.optimizers.zero_grad_all()

            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            # this will only be RGB loss / pose delta regularization during eval
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())

            if "camera_opt" in self.optimizers.optimizers:
                if not torch.isfinite(eval_loss):
                    raise ValueError(f"Eval Loss is not finite: {eval_loss_dict}.")

                eval_loss_dict["main_loss"].backward()  # type: ignore
                # fine-tune eval poses when doing training camera optimization
                self.optimizers.optimizer_step("camera_opt")
                if "object_poses" in self.optimizers.optimizers.keys():
                    self.optimizers.optimizer_step("object_poses")

            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
