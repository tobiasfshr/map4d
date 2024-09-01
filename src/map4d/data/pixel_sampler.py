"""Pixel Sampler containing some fixes for varying input resolutions."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type, Union

import torch
from jaxtyping import Int
from nerfstudio.data.pixel_samplers import PixelSampler as VanillaPixelSampler
from nerfstudio.data.pixel_samplers import PixelSamplerConfig as VanillaPixelSamplerConfig
from torch import Tensor


@dataclass
class PixelSamplerConfig(VanillaPixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    image_like_keys: Tuple[str, ...] = ("image", "depth_image")
    """Keys that are image like and should be sampled from."""


class PixelSampler(VanillaPixelSampler):
    """Pixel Sampler class with better support for varying input resolutions."""

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Modified to avoid .nonzero() call on the mask tensor, which is very slow.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor):
            sampled_indices = []
            while sum([len(i) for i in sampled_indices]) < batch_size:
                indices = torch.floor(
                    torch.rand((batch_size, 3), device=device)
                    * torch.tensor([num_images, image_height, image_width], device=device)
                ).long()
                keep_mask = mask[indices[:, 0], indices[:, 1], indices[:, 2]].squeeze(-1)
                indices = indices[keep_mask]
                sampled_indices.append(indices)

            indices = torch.cat(sampled_indices)[:batch_size]
        else:
            indices = (
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Modified to support, e.g., depth images and other image like tensors.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        image_like_fields = {key: [] for key in self.config.image_like_keys if key in batch}

        if "mask" in batch:
            num_rays_in_batch = max(1, num_rays_per_batch // num_images)
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                for key in image_like_fields:
                    if key in batch:
                        image_like_fields[key].append(batch[key][i][indices[:, 1], indices[:, 2]])
        else:
            num_rays_in_batch = max(1, num_rays_per_batch // num_images)
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(
                        num_rays_in_batch, 1, image_height, image_width, device=device
                    )
                else:
                    indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                for key in image_like_fields:
                    if key in batch:
                        image_like_fields[key].append(batch[key][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in ["image_idx", "mask", *image_like_fields.keys()] and value is not None
        }

        for key, value in image_like_fields.items():
            collated_batch[key] = torch.cat(value, dim=0)
            assert (
                collated_batch[key].shape[0] == num_rays_per_batch
            ), f"key: {key}, shape: {collated_batch[key].shape}, num_rays_per_batch: {num_rays_per_batch}"

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
