from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, ForwardRef, Generic, Tuple, Type, cast, get_args, get_origin

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from torch.utils.data import DataLoader, DistributedSampler

from map4d.data.dataset import CameraImageDataset

# to avoid issues with open file limit
torch.multiprocessing.set_sharing_strategy("file_system")


def collate_fn(batch):
    """Collate function for the dataloader"""
    return batch[0]


@dataclass
class MultiProcessFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: MultiProcessFullImageDatamanager)
    """Target class to instantiate"""


class MultiProcessFullImageDatamanager(FullImageDatamanager, Generic[TDataset]):
    """FullImageDatamanager using torch multiprocess dataloading."""

    def __init__(self, config: MultiProcessFullImageDatamanagerConfig, *args, **kwargs):
        self._fixed_indices_eval_dataloader = None
        super().__init__(config, *args, **kwargs)

    def cache_images(self, cache_images):
        return [], []

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[MultiProcessFullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is MultiProcessFullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is MultiProcessFullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is MultiProcessFullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def setup_train(self):
        """Sets up the data loaders for training"""
        self.train_image_dataset = CameraImageDataset(self.train_dataset)
        if self.world_size > 0:
            self.train_sampler = DistributedSampler(self.train_image_dataset, self.world_size, self.local_rank)
            self.train_dataloader = DataLoader(
                self.train_image_dataset,
                batch_size=1,
                sampler=self.train_sampler,
                num_workers=2,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_image_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=2,
                persistent_workers=True,
                collate_fn=collate_fn,
            )

        self.iter_train_dataloader = iter(self.train_dataloader)

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        self.eval_image_dataset = CameraImageDataset(self.eval_dataset)
        if self.world_size > 0:
            self.eval_sampler = DistributedSampler(self.eval_image_dataset, self.world_size, self.local_rank)
            self.eval_dataloader = DataLoader(
                self.eval_image_dataset,
                batch_size=1,
                sampler=self.eval_sampler,
                num_workers=1,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
        else:
            self.eval_dataloader = DataLoader(
                self.eval_image_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=1,
                persistent_workers=True,
                collate_fn=collate_fn,
            )

        self.iter_eval_dataloader = iter(self.eval_dataloader)

    @property
    def fixed_indices_eval_dataloader(self):
        """Return dataloader for evaluation with fixed indices."""
        if self._fixed_indices_eval_dataloader is None:
            if self.world_size > 0:
                eval_sampler = DistributedSampler(
                    self.eval_image_dataset, self.world_size, self.local_rank, shuffle=False
                )
                eval_dataloader = DataLoader(
                    self.eval_image_dataset,
                    batch_size=1,
                    sampler=eval_sampler,
                    num_workers=1,
                    persistent_workers=True,
                    collate_fn=collate_fn,
                )
            else:
                eval_dataloader = DataLoader(
                    self.eval_image_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    persistent_workers=True,
                    collate_fn=collate_fn,
                )
            self._fixed_indices_eval_dataloader = eval_dataloader
        return iter(self._fixed_indices_eval_dataloader)

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        result = next(self.iter_train_dataloader, None)
        if result is None:
            self.iter_train_dataloader = iter(self.train_dataloader)
            result = next(self.iter_train_dataloader)
        camera, data = result
        camera = camera.to(self.device)
        data["image"] = data["image"].to(self.device)
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        result = next(self.iter_eval_dataloader, None)
        if result is None:
            self.iter_eval_dataloader = iter(self.eval_dataloader)
            result = next(self.iter_eval_dataloader)
        camera, data = result
        camera = camera.to(self.device)
        data["image"] = data["image"].to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        result = next(self.iter_eval_dataloader, None)
        if result is None:
            self.iter_eval_dataloader = iter(self.eval_dataloader)
            result = next(self.iter_eval_dataloader)
        camera, data = result
        camera = camera.to(self.device)
        data["image"] = data["image"].to(self.device)
        return camera, data
