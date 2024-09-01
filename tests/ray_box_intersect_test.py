import math

import pytest
import torch
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

from map4d.model.util import calculate_local_ray_samples, get_objects_per_ray


def rotate_yaw(pts: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate points around the z-axis.

    Args:
        pts: Tensor of shape (..., 3).
        yaw: Rotation angle in radians.

    Returns:
        Tensor of shape (..., 3).
    """
    c, s = torch.cos(yaw), torch.sin(yaw)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pts.dtype, device=pts.device)
    return torch.einsum("...ij,...j->...i", R, pts)


def world2object(positions, center, yaw):
    positions = positions - center
    positions = rotate_yaw(positions, -yaw)
    return positions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_ray_box_intersect():
    """Test ray_box_intersect, get_objects_per_ray and calculate_local_ray_samples functions."""
    from map4d.cuda import ray_box_intersect

    ray_bundle = RayBundle(
        torch.tensor([[0.0, 0.0, 0.0]]).cuda(),
        torch.tensor([[0.0, 0.0, 1.0]]).cuda(),
        torch.tensor([[1.0]]).cuda(),
        times=torch.tensor([[-1.0]]).cuda(),
        metadata={"sequence_ids": torch.tensor([[0.0]]).cuda()},
    )
    max_depth, num_bins = 10.0, 10
    bin_starts = torch.linspace(0.0, 1.0, num_bins).cuda() * max_depth - 1e-6
    bin_ends = torch.linspace(0.0, 1.0, num_bins).cuda() * max_depth + (max_depth / num_bins) - 1e-6
    ray_samples = ray_bundle.get_ray_samples(bin_starts[:, None], bin_ends[:, None])
    # N_times
    seq_ids = torch.tensor([0.0, 0.0]).cuda()
    times = torch.tensor([-1.01, -0.98]).cuda()
    # N_times, N_objects
    poses = torch.tensor(
        [
            [[0.0, 0.0, 5.0, math.pi / 4], [0.0, 0.0, 10.0, math.pi / 2], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 5.0, math.pi / 4], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    ).cuda()
    dims = torch.tensor(
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    ).cuda()
    obj_ids = torch.tensor([[2.0, 0.0, -1.0], [2.0, -1.0, -1.0]]).cuda()

    per_ray_boxes3d, per_ray_obj_ids = get_objects_per_ray(
        ray_bundle.metadata["sequence_ids"], ray_bundle.times, seq_ids, times, obj_ids, dims, poses
    )
    assert torch.isclose(
        torch.tensor([2.0, 0.0, -1.0]).cuda(), per_ray_obj_ids
    ).all(), f"per_ray_obj_ids: {per_ray_obj_ids}, expected: {torch.tensor([2.0, 0.0, -1.0]).cuda()}"

    local_origins, local_directions, near_fars, _ = ray_box_intersect(
        ray_bundle.origins, ray_bundle.directions, per_ray_boxes3d
    )

    obj_ids, boxes, local_origins, local_directions, starts, ends, hit_mask = calculate_local_ray_samples(
        ray_samples.frustums.starts,
        ray_samples.frustums.ends,
        per_ray_boxes3d,
        per_ray_obj_ids,
        local_origins,
        local_directions,
        near_fars,
    )
    local_ray_samples = RaySamples(
        frustums=Frustums(
            local_origins, local_directions, starts, ends, pixel_area=ray_samples.frustums.pixel_area[hit_mask]
        )
    )
    ray_positions = ray_samples.frustums.get_positions()[hit_mask]
    local_positions = local_ray_samples.frustums.get_positions()
    assert torch.isclose(
        torch.tensor([2, 0], dtype=torch.long).cuda(), obj_ids
    ).all(), f"obj_ids: {obj_ids}, expected: {torch.tensor([2, 0], dtype=torch.long).cuda()}"
    for pos, box, local_pos in zip(ray_positions, boxes, local_positions):
        pos = world2object(pos, box[:3], box[6])
        assert torch.isclose(
            pos, local_pos
        ).all(), f"local_positions not equal to ray_positions, expected: {pos}, got: {local_pos}"
