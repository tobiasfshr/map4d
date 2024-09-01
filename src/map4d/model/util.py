from typing import Tuple

import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.poses import to4x4
from torch import Tensor

from map4d.common.geometry import inverse_rigid_transform
from map4d.common.visualize import boxes3d_to_corners3d

if torch.distributed.is_available():
    from torch.distributed.utils import (
        _verify_param_shape_across_processes,
        _sync_module_states,
    )

from map4d.common.geometry import angle_lerp


# update DDP reducer to register hooks for replaced parameters
def trigger_reducer_update(model: torch.nn.parallel.DistributedDataParallel) -> None:
    # Build parameters for reducer.
    parameters, expect_sparse_gradient = model._build_params_for_reducer()
    # Verify model equivalence.
    _verify_param_shape_across_processes(model.process_group, parameters)
    # Sync params and buffers. Ensures all DDP models start off at the same value.
    _sync_module_states(
        module=model.module,
        process_group=model.process_group,
        broadcast_bucket_size=model.broadcast_bucket_size,
        src=0,
        params_and_buffers_to_ignore=model.parameters_to_ignore,
    )
    # In debug mode, build a mapping of parameter index -> parameter.
    param_to_name_mapping = model._build_debug_param_to_name_mapping(parameters)
    # Builds reducer.
    model._ddp_init_helper(
        parameters,
        expect_sparse_gradient,
        param_to_name_mapping,
        False,
    )


@torch.jit.script
def get_objects_per_ray(
    seq_ids: Tensor,
    times: Tensor,
    object_seq_ids: Tensor,
    object_times: Tensor,
    object_ids: Tensor,
    object_dims: Tensor,
    object_poses: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Get object ids and boxes per ray.

    Args:
        seq_ids (Tensor): N_rays, 1
        times (Tensor): N_rays, 1
        object_seq_ids (Tensor): N_seq * N_times
        object_times (Tensor): N_seq * N_times
        object_ids (Tensor): N_seq * N_times, N_boxes
        object_dims (Tensor): N_seq * N_times, N_boxes, 3
        object_poses (Tensor): N_seq * N_times, N_boxes, 4

    Returns:
        Tuple[Tensor, Tensor]:
            per_ray_boxes3d: N_rays, N_boxes, 7
            per_ray_obj_ids: N_rays, N_boxes
    """
    # match ray seq ids / times to object seq ids / times
    seq_mask = seq_ids == object_seq_ids[None]  # N_rays, (N_times * N_seq)
    time_diffs = (times - object_times).abs()  # N_rays, (N_times * N_seq)
    time_diffs[~seq_mask] = float("inf")

    # interpolate poses at closest t0, t1 to continuous ray time t
    time_diffs, ray_obj_idcs = torch.topk(time_diffs, 2, dim=1, largest=False)
    blend_weights = 1.0 - (time_diffs / (time_diffs.sum(dim=1, keepdim=True)))

    # get object info at t0 and t1
    # t0: index with timestep
    per_ray_obj_ids = object_ids[ray_obj_idcs[:, 0]]
    per_ray_obj_dims = object_dims[ray_obj_idcs[:, 0]]
    per_ray_obj_poses = object_poses[ray_obj_idcs[:, 0]]
    # t1: match t0 ids to t1 ids with t0 pose as default if no match
    per_ray_obj_ids_t1 = per_ray_obj_ids.unsqueeze(-1) == object_ids[ray_obj_idcs[:, 1]].unsqueeze(1)
    per_ray_obj_ids_t1[per_ray_obj_ids == -1] = False
    per_ray_obj_ids_t1 = per_ray_obj_ids_t1.nonzero()
    per_ray_obj_poses_t1 = per_ray_obj_poses.clone()
    per_ray_obj_poses_t1[per_ray_obj_ids_t1[:, 0], per_ray_obj_ids_t1[:, 1]] = object_poses[ray_obj_idcs[:, 1]][
        per_ray_obj_ids_t1[:, 0], per_ray_obj_ids_t1[:, 2]
    ]

    # interpolate to time t: N_rays, N_boxes, 4
    per_ray_obj_poses = torch.cat(
        (
            torch.lerp(per_ray_obj_poses[..., :3], per_ray_obj_poses_t1[..., :3], blend_weights[..., None, 1:2]),
            angle_lerp(per_ray_obj_poses[..., 3:4], per_ray_obj_poses_t1[..., 3:4], blend_weights[..., None, 1:2]),
        ),
        -1,
    )
    per_ray_boxes3d = torch.cat([per_ray_obj_poses, per_ray_obj_dims], -1)[:, :, [0, 1, 2, 4, 5, 6, 3]]
    return per_ray_boxes3d, per_ray_obj_ids


@torch.jit.script
def calculate_local_ray_samples(
    ray_sample_starts: Tensor,
    ray_sample_ends: Tensor,
    per_ray_boxes3d: Tensor,
    per_ray_obj_ids: Tensor,
    local_origins: Tensor,
    local_directions: Tensor,
    near_fars: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Calculate local ray inputs for the dynamic field.

    Args:
        ray_sample_starts (Tensor): N_rays, N_samples, 1 start of bins
        ray_sample_ends (Tensor):  N_rays, N_samples, 1 end of bins
        per_ray_boxes3d (Tensor): N_rays, N_boxes, 7
        per_ray_obj_ids (Tensor): N_rays, N_boxes
        local_origins (Tensor): N_rays, N_boxes, 3
        local_directions (Tensor): N_rays, N_boxes, 3
        near_fars (Tensor): N_rays, N_boxes, 2

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            obj_ids: N_rays, N_samples, 1
            obj_boxes: N_rays, N_samples, 7
            local_origins: N_rays, N_samples, 3
            local_directions: N_rays, N_samples, 3
            local_starts: N_rays, N_samples, 1
            local_ends: N_rays, N_samples, 1
            ray_sample_hit_mask: N_rays, N_samples, 1
    """
    N_rays, N_boxes, _ = per_ray_boxes3d.shape
    N_samples = ray_sample_starts.shape[1]

    # Expand ray starts / ends from  N_rays, N_samples, 1 to N_rays, N_samples, N_boxes, compare with near_fars
    ray_sample_starts = ray_sample_starts.expand(N_rays, N_samples, N_boxes)
    ray_sample_ends = ray_sample_ends.expand(N_rays, N_samples, N_boxes)
    depth_values = (ray_sample_starts + ray_sample_ends) / 2
    sample_hit_mask = (depth_values >= near_fars[:, None, :, 0]) & (depth_values <= near_fars[:, None, :, 1])

    # # constrain a single point to be within a single 3D box, i.e. get index of first hit, set all other to false
    ray_sample_hit_mask, first_sample_hit = torch.max(sample_hit_mask, dim=2)
    sample_hit_mask = torch.zeros_like(sample_hit_mask)
    sample_hit_mask.scatter_(2, first_sample_hit.unsqueeze(2), ray_sample_hit_mask.unsqueeze(2))

    # calculate local origins, directions
    ray_sample_starts = ray_sample_starts[sample_hit_mask]
    ray_sample_ends = ray_sample_ends[sample_hit_mask]
    local_origins = local_origins.unsqueeze(1).expand(N_rays, N_samples, N_boxes, 3)[sample_hit_mask]
    local_directions = local_directions.unsqueeze(1).expand(N_rays, N_samples, N_boxes, 3)[sample_hit_mask]

    obj_ids = per_ray_obj_ids.unsqueeze(1).expand(N_rays, N_samples, N_boxes)
    obj_ids = obj_ids[sample_hit_mask].long()
    obj_boxes = per_ray_boxes3d.unsqueeze(1).expand(N_rays, N_samples, N_boxes, 7)
    obj_boxes = obj_boxes[sample_hit_mask]

    # shapenet cars are usually within [-0.5, 0.5], thus normalize with 1.0 * max(dim)
    normalizer = obj_boxes[:, 3:6].max(dim=-1, keepdim=True)[0]
    local_origins /= normalizer
    ray_sample_starts = ray_sample_starts.unsqueeze(-1) / normalizer
    ray_sample_ends = ray_sample_ends.unsqueeze(-1) / normalizer

    return obj_ids, obj_boxes, local_origins, local_directions, ray_sample_starts, ray_sample_ends, ray_sample_hit_mask


@torch.jit.script
def get_objects_at_time(
    object_poses: Tensor,
    object_ids: Tensor,
    object_dims: Tensor,
    object_times: Tensor,
    object_seq_ids: Tensor,
    object_class_ids: Tensor,
    time: Tensor,
    sequence: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Get object ids, poses, dims at time t. Interpolates between t0 and t1.

    Args:
        object_poses (Tensor): N_seq * N_times, N_boxes, 4
        object_ids (Tensor): N_seq * N_times, N_boxes
        object_dims (Tensor): N_seq * N_times, N_boxes, 3
        object_times (Tensor): N_seq * N_times
        object_seq_ids (Tensor): N_seq * N_times
        object_class_ids (Tensor): N_seq * N_times
        time (Tensor): scalar time to get objects at
        sequence (Tensor): scalar int sequence to get objects at

    Returns:
        tuple[Tensor, Tensor, Tensor]: object ids, poses, dims, classes
    """
    time_diffs = (time - object_times).abs()  # (N_times * N_seq)
    seq_mask = object_seq_ids == sequence  # (N_times * N_seq)
    time_diffs[~seq_mask] = float("inf")

    # interpolate poses at closest t0, t1 to continuous time t
    time_diffs, time_idcs = torch.topk(time_diffs, 2, dim=0, largest=False)
    blend_weights = 1.0 - (time_diffs / (time_diffs.sum() + 1e-8))

    # get object info at t0 and t1
    # t0: index with timestep
    t0_obj_ids = object_ids[time_idcs[0]]
    t0_obj_poses = object_poses[time_idcs[0]]
    # don't interpolate dims / classes because they do not change over time
    t0_obj_dims = object_dims[time_idcs[0]]
    t0_obj_classes = object_class_ids[time_idcs[0]]

    # t1: match t0 ids to t1 ids with t0 pose as default if no match
    t1_obj_ids = t0_obj_ids.unsqueeze(-1) == object_ids[time_idcs[1]].unsqueeze(0)
    t1_obj_ids[t0_obj_ids == -1] = False
    t1_obj_ids = t1_obj_ids.nonzero()
    t1_obj_poses = t0_obj_poses.clone()
    t1_obj_poses[t1_obj_ids[:, 0]] = object_poses[time_idcs[1]][t1_obj_ids[:, 1]]

    # stack and interpolate to time t: 2, N_boxes, 4 -> N_boxes, 4
    t_obj_poses = torch.cat(
        (
            torch.lerp(t0_obj_poses[..., :3], t1_obj_poses[..., :3], blend_weights[1].unsqueeze(-1)),
            angle_lerp(t0_obj_poses[..., 3:4], t1_obj_poses[..., 3:4], blend_weights[1].unsqueeze(-1)),
        ),
        -1,
    )

    mask = t0_obj_ids != -1
    return t0_obj_ids[mask].long(), t_obj_poses[mask], t0_obj_dims[mask], t0_obj_classes[mask].long()


def c2w_to_local(camera2world: Tensor, obj_poses: Tensor, obj_dims_max: Tensor) -> Tensor:
    """Clone, detach, transform camera2world to local object frame.

    Args:
        camera2world (Tensor): N, 3, 4
        obj_poses (Tensor): N, 4
        obj_dims_max (Tensor): N, 1 maximum of xyz dimensions used for normalization

    Returns:
        Tensor: N, 3, 4 local transformations
    """
    c2w = camera2world.clone().detach()
    c2w[..., :3, 3] -= obj_poses[..., :3]
    yaw = -obj_poses[..., 3]
    cos, sin, zeros, ones = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    rotation = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], -1).view(*yaw.shape, 3, 3)
    c2w[..., :3, 3:4] = torch.matmul(rotation, c2w[..., :3, 3:4])
    c2w[..., :3, 3] /= obj_dims_max
    c2w[..., :3, :3] = torch.matmul(rotation, c2w[..., :3, :3])
    return c2w


def mask_images(batch, gt_rgb, predicted_rgb, size_divisor=1):
    if "mask" in batch:
        mask = batch["mask"]
        if size_divisor > 1:
            newsize = [mask.shape[0] // size_divisor, mask.shape[1] // size_divisor]
            mask = F.interpolate(mask.permute(2, 0, 1).unsqueeze(0), newsize, mode="nearest")
            mask = mask.squeeze(0).permute(1, 2, 0)
        mask = mask.squeeze(-1)
        gt_rgb[mask == False] = 0  # noqa: E712
        predicted_rgb[mask == False] = 0  # noqa: E712
    return gt_rgb, predicted_rgb


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return F.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


def opengl_frustum_check(obj_poses: Tensor, obj_dims: Tensor, camera: Cameras) -> Tensor:
    """Check if 3D boxes are in view frustum."""
    # use a buffer (+ 2x object height) for shadows and alike
    obj_dims_buffer = obj_dims.clone()
    obj_dims_buffer[:, :-1] += obj_dims[:, -1:] * 2.0
    corners = boxes3d_to_corners3d(obj_poses, obj_dims_buffer)

    # world to camera
    w2c = inverse_rigid_transform(to4x4(camera.camera_to_worlds[0]))
    corners = corners @ w2c[:3, :3].T + w2c[:3, 3]

    lim_x_pos = (camera.width - camera.cx) / camera.fx
    lim_x_neg = -camera.cx / camera.fx
    lim_y_pos = (camera.height - camera.cy) / camera.fy
    lim_y_neg = -camera.cy / camera.fy

    tx, ty, tz = corners[..., 0], corners[..., 1], corners[..., 2]
    view_mask = torch.ones_like(corners[..., 0]).bool()
    view_mask &= tx / tz > lim_x_neg
    view_mask &= tx / tz < lim_x_pos
    view_mask &= ty / tz > lim_y_neg
    view_mask &= ty / tz < lim_y_pos
    view_mask &= tz <= 0
    return view_mask.any(dim=-1)
