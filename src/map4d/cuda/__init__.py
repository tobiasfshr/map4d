from typing import Any, Tuple

import torch
from torch import Tensor, nn

from . import ray_box_intersect_cuda, video_embedding_cuda


class VideoEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, times: torch.Tensor, video_ids: torch.Tensor, weights: torch.Tensor, num_frequencies: int
    ) -> torch.Tensor:
        embeddings = video_embedding_cuda.video_embedding_forward(times, video_ids, weights, num_frequencies)
        ctx.save_for_backward(times, video_ids, torch.IntTensor([weights.shape[0], num_frequencies]))
        return embeddings

    @staticmethod
    def backward(ctx: Any, d_loss_embedding: torch.Tensor):
        times, video_ids, num_sequences_and_frequencies = ctx.saved_tensors
        d_loss_weights = video_embedding_cuda.video_embedding_backward(
            d_loss_embedding.contiguous(),
            times,
            video_ids,
            num_sequences_and_frequencies[0].item(),
            num_sequences_and_frequencies[1].item(),
        )

        return None, None, d_loss_weights, None


class VideoEmbedding(nn.Module):
    def __init__(self, num_videos: int, num_frequencies: int, embedding_dim: int):
        super(VideoEmbedding, self).__init__()
        self.num_videos = num_videos
        self.num_frequencies = num_frequencies
        self.sequence_code_weights = nn.Parameter(
            torch.empty(size=(num_videos, embedding_dim, num_frequencies * 2 + 1), dtype=torch.float32),
            requires_grad=True,
        )
        torch.nn.init.normal_(self.sequence_code_weights)

    def forward(self, times: torch.Tensor, video_ids: torch.Tensor) -> torch.Tensor:
        times_orignal_shape = times.shape
        times = times.contiguous().view(-1)
        video_ids = video_ids.int().contiguous().view(-1)
        return VideoEmbeddingFunction.apply(times, video_ids, self.sequence_code_weights, self.num_frequencies).view(
            (*times_orignal_shape[:-1], -1)
        )


@torch.no_grad()
def ray_box_intersect(
    rays_o: Tensor,
    rays_d: Tensor,
    boxes3d: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Ray-Box intersection.

    Args:
        origins: (N, 3) global origins [num_rays, 3]
        directions: (N, 3)
        boxes3d: (N, M, 7), [x, y, z, dx, dy, dz, heading] set of boxes depending on ray seq id / time

    Returns:
        local_origins: (N, M, 3) in local box coordinate
        local_directions: (N, M, 3) in local box coordinate
        t_min_max: (N, M, 2) tmin, tmax for each ray-box pair.
        box_hit_mask: (N, M) indices of the box, -1 if no intersection
    """
    assert rays_o.ndim == 2 and rays_o.shape[-1] == 3
    assert rays_d.ndim == 2 and rays_d.shape[-1] == 3
    assert boxes3d.ndim == 3 and boxes3d.shape[-1] == 7

    num_rays, num_boxes = rays_o.shape[0], boxes3d.shape[1]

    local_origins = rays_o.new_zeros((num_rays, num_boxes, 3))
    local_directions = rays_o.new_zeros((num_rays, num_boxes, 3))
    near_fars = rays_o.new_zeros((num_rays, num_boxes, 2))
    box_hit_mask = rays_o.new_zeros((num_rays, num_boxes), dtype=torch.bool)

    ray_box_intersect_cuda.forward(
        rays_o.contiguous(),
        rays_d.contiguous(),
        boxes3d.contiguous(),
        local_origins,
        local_directions,
        near_fars,
        box_hit_mask,
    )
    return local_origins, local_directions, near_fars, box_hit_mask
