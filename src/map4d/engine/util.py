"""Trainer utils"""
import math
from typing import Literal


def scale_lr(lr: float, world_size: int, method: Literal["none", "linear", "sqrt"]):
    """Scale learning rate based on world size."""
    if method == "none":
        return lr
    if method == "linear":
        return lr * world_size
    if method == "sqrt":
        return lr * (world_size**0.5)
    raise ValueError(f"Unknown lr scaling method: {method}")


def scale_by_world_size(config, world_size: int):
    """Scale config based on world size. Divides steps by world size and scales learning rates."""
    scale_lr_method = config.scale_lrs
    # scale learning rates
    for optimizer in config.optimizers.values():
        for key, value in optimizer.items():
            if key == "optimizer":
                value.lr = scale_lr(value.lr, world_size, scale_lr_method)
            if key == "scheduler" and value is not None:
                value.lr_final = scale_lr(value.lr_final, world_size, scale_lr_method)
                value.max_steps = math.ceil(value.max_steps / world_size)

    # scale number of steps (divide by world size)
    config.max_num_iterations = math.ceil(config.max_num_iterations / world_size)
    config.steps_per_eval_batch = math.ceil(config.steps_per_eval_batch / world_size)
    config.steps_per_eval_image = math.ceil(config.steps_per_eval_image / world_size)
    config.steps_per_save = math.ceil(config.steps_per_save / world_size)
    config.steps_per_eval_all_images = math.ceil(config.steps_per_eval_all_images / world_size)

    # scale step-based model params: refine_every, warmup_length, stop_split_at, stop_screen_size_at
    if hasattr(config.pipeline.model, "refine_every"):
        config.pipeline.model.refine_every = math.ceil(config.pipeline.model.refine_every / world_size)
    if hasattr(config.pipeline.model, "warmup_length"):
        config.pipeline.model.warmup_length = math.ceil(config.pipeline.model.warmup_length / world_size)
    if hasattr(config.pipeline.model, "stop_split_at"):
        config.pipeline.model.stop_split_at = math.ceil(config.pipeline.model.stop_split_at / world_size)
    if hasattr(config.pipeline.model, "stop_screen_size_at"):
        config.pipeline.model.stop_screen_size_at = math.ceil(config.pipeline.model.stop_screen_size_at / world_size)
    return config
