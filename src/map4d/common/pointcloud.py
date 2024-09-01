"""Pointcloud processing utilities."""
import torch
from torch import Tensor

from map4d.common.geometry import rotate_z


def voxelize_pointcloud(
    pointcloud: Tensor, features: Tensor | None = None, voxel_size: float = 0.25, return_counts: bool = False
) -> Tensor:
    """Voxelizes a pointcloud by calculating the mean position of points in each voxel."""
    assert pointcloud.shape[-1] == 3, f"Pointcloud must have shape (N, 3), got {pointcloud.shape}"
    assert features is None or features.shape[0] == pointcloud.shape[0], "Features must have same length as pointcloud"
    if voxel_size <= 0:
        if features is not None:
            if return_counts:
                return pointcloud, features, torch.ones_like(pointcloud[..., 0])
            return pointcloud, features
        if return_counts:
            return pointcloud, torch.ones_like(pointcloud[..., 0])
        return pointcloud

    # Calculate the voxel indices for each point
    voxel_indices = (pointcloud / voxel_size).floor().long()

    # Calculate the unique voxel indices to determine the number of voxels
    unique_voxel_indices, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)

    # Create a tensor to accumulate point positions and count in each voxel
    voxel_positions = torch.zeros((unique_voxel_indices.shape[0], 3), device=pointcloud.device)
    voxel_counts = torch.zeros((unique_voxel_indices.shape[0],), device=pointcloud.device)

    # Scatter the point positions into the corresponding voxel
    voxel_positions.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), pointcloud)
    voxel_counts.scatter_add_(0, inverse_indices, torch.ones_like(pointcloud[..., 0]))

    # Calculate the mean position for each voxel
    voxel_means = voxel_positions / voxel_counts.float().unsqueeze(-1)

    if features is not None:
        voxel_features = torch.zeros((unique_voxel_indices.shape[0], features.shape[-1]), device=pointcloud.device)
        voxel_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.shape[-1]), features)
        voxel_features = voxel_features / voxel_counts.float().unsqueeze(-1)
        if return_counts:
            return voxel_means, voxel_features, voxel_counts
        return voxel_means, voxel_features
    if return_counts:
        return voxel_means, voxel_counts
    return voxel_means


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Applies transform to points.

    Args:
        points (Tensor): points of shape (N, D) or (B, N, D).
        transform (Tensor): transforms of shape (D+1, D+1) or (B, D+1, D+1).

    Returns:
        Tensor: (N, D) / (B, N, D) transformed points.

    Raises:
        ValueError: Either points or transform have incorrect shape
    """
    hom_coords = torch.cat([points, torch.ones_like(points[..., 0:1])], -1)
    if len(points.shape) == 2:
        if len(transform.shape) == 3:
            assert transform.shape[0] == 1, "Got multiple transforms for single point set!"
            transform = transform.squeeze(0)
        transform = transform.T
    elif len(points.shape) == 3:
        if len(transform.shape) == 2:
            transform = transform.T.unsqueeze(0)
        elif len(transform.shape) == 3:
            transform = transform.permute(0, 2, 1)
        else:
            raise ValueError(f"Shape of transform invalid: {transform.shape}")
    else:
        raise ValueError(f"Shape of input points invalid: {points.shape}")
    points_transformed = hom_coords @ transform
    return points_transformed[..., : points.shape[-1]]


def get_points_in_boxes3d(points: Tensor, box_params: Tensor) -> tuple[Tensor, list[Tensor]]:
    """Get points inside a set of 3D boxes.

    If there are collisions, the first box that the point is inside is chosen.

    Args:
        points (Tensor): [N, 3] points.
        box_params (Tensor): [M, 7] 3D boxes.

    Returns:
        tuple[Tensor, list[Tensor]]: [N,] index of box (-1 if none), list of normalized points inside boxes, length M.
    """
    points_inside_boxes = torch.ones_like(points[:, 0], dtype=torch.long) * -1
    points_per_box = []
    for i, box_param in enumerate(box_params):
        assert box_param.shape == (7,), f"Box param shape must be (7,), got {box_param.shape}"
        points_normalized = points[points_inside_boxes == -1] - box_param[0:3]
        points_normalized = rotate_z(points_normalized, -box_param[-1])
        points_inside_box = torch.all(
            torch.logical_and(points_normalized < box_param[3:6] / 2, points_normalized > -box_param[3:6] / 2), dim=1
        )
        points_per_box.append(points_normalized[points_inside_box])
        mask = points_inside_boxes == -1
        mask[mask == True] = points_inside_box  # noqa: E712
        points_inside_boxes[mask] = i
    return points_inside_boxes, points_per_box
