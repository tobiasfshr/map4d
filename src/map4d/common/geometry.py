"""Geometry related functions. Quaternion functions are copied from PyTorch3D."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

opencv_to_opengl = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def inverse_pinhole(intrinsic_matrix: Tensor) -> Tensor:
    """Calculate inverse of pinhole projection matrix.

    Args:
        intrinsic_matrix (Tensor): [..., 3, 3] intrinsics or single [3, 3]
            intrinsics.

    Returns:
        Tensor:  Inverse of input intrinisics.
    """
    squeeze = False
    inv = intrinsic_matrix.clone()
    if len(intrinsic_matrix.shape) == 2:
        inv = inv.unsqueeze(0)
        squeeze = True

    inv[..., 0, 0] = 1.0 / inv[..., 0, 0]
    inv[..., 1, 1] = 1.0 / inv[..., 1, 1]
    inv[..., 0, 2] = -inv[..., 0, 2] * inv[..., 0, 0]
    inv[..., 1, 2] = -inv[..., 1, 2] * inv[..., 1, 1]

    if squeeze:
        inv = inv.squeeze(0)
    return inv


def points_inside_image(
    points_coord: torch.Tensor,
    depths: torch.Tensor,
    images_hw: torch.Tensor | tuple[int, int],
) -> torch.Tensor:
    """Generate binary mask.

    Creates a mask that is true for all point coordiantes that lie inside the
    image,

    Args:
        points_coord (torch.Tensor): 2D pixel coordinates of shape [..., 2].
        depths (torch.Tensor): Associated depth of each 2D pixel coordinate.
        images_hw:  (torch.Tensor| tuple[int, int]]) Associated tensor of image
                    dimensions, shape [..., 2] or single height, width pair.

    Returns:
        torch.Tensor: Binary mask of points inside an image.
    """
    mask = torch.ones_like(depths)
    h: int | torch.Tensor
    w: int | torch.Tensor

    if isinstance(images_hw, tuple):
        h, w = images_hw
    else:
        h, w = images_hw[..., 0], images_hw[..., 1]
    mask = torch.logical_and(mask, depths > 0)
    mask = torch.logical_and(mask, points_coord[..., 0] > 0)
    mask = torch.logical_and(mask, points_coord[..., 0] < w - 1)
    mask = torch.logical_and(mask, points_coord[..., 1] > 0)
    mask = torch.logical_and(mask, points_coord[..., 1] < h - 1)
    return mask


def project_points(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project points to pixel coordinates with given intrinsics.

    Args:
        points: (N, 3) or (B, N, 3) 3D coordinates.
        intrinsics: (3, 3) or (B, 3, 3) intrinsic camera matrices.

    Returns:
        torch.Tensor: (N, 2) or (B, N, 2) 2D pixel coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    assert points.shape[-1] == 3, "Input coordinates must be 3 dimensional!"
    hom_coords = points / points[..., 2:3]
    if len(hom_coords.shape) == 2:
        assert len(intrinsics.shape) == 2, "Got multiple intrinsics for single point set!"
        intrinsics = intrinsics.T
    elif len(hom_coords.shape) == 3:
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics.unsqueeze(0)
        intrinsics = intrinsics.permute(0, 2, 1)
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    pts_2d = hom_coords @ intrinsics
    return pts_2d[..., :2]


def generate_depth_map(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Generate depth map for given pointcloud.

    Args:
        points: (N, 3) coordinates.
        intrinsics: (3, 3) intrinsic camera matrices.
        image_hw: (tuple[int,int]) height, width of the image

    Returns:
        torch.Tensor: Projected depth map of the given pointcloud.
                      Invalid depth has 0 values
    """
    pts_2d = project_points(points, intrinsics).round()
    depths = points[:, 2]
    depth_map = points.new_zeros(image_hw)
    mask = points_inside_image(pts_2d, depths, image_hw)
    pts_2d = pts_2d[mask].long()
    depth_map[pts_2d[:, 1], pts_2d[:, 0]] = depths[mask]
    return depth_map


def unproject_points(points: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Un-projects pixel coordinates to 3D coordinates with given intrinsics.

    Args:
        points: (N, 2) or (B, N, 2) 2D pixel coordinates.
        depths: (N,) / (N, 1) or (B, N,) / (B, N, 1) depth values.
        intrinsics: (3, 3) or (B, 3, 3) intrinsic camera matrices.

    Returns:
        torch.Tensor: (N, 3) or (B, N, 3) 3D coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    if len(points.shape) == 2:
        assert len(intrinsics.shape) == 2 or intrinsics.shape[0] == 1, "Got multiple intrinsics for single point set!"
        if len(intrinsics.shape) == 3:
            intrinsics = intrinsics.squeeze(0)
        inv_intrinsics = inverse_pinhole(intrinsics).transpose(0, 1)
        if len(depths.shape) == 1:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 2, "depths must have same dims as points"
    elif len(points.shape) == 3:
        inv_intrinsics = inverse_pinhole(intrinsics).transpose(-2, -1)
        if len(depths.shape) == 2:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 3, "depths must have same dims as points"
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    hom_coords = torch.cat([points, torch.ones_like(points)[..., 0:1]], -1)
    pts_3d = hom_coords @ inv_intrinsics
    pts_3d *= depths
    return pts_3d


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates=True,
    device=torch.device("cpu"),
) -> torch.Tensor:
    """Generates a coordinate grid for an image.
    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample
    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.
    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys], indexing="ij")).permute(2, 1, 0).contiguous()
    return base_grid


def depth_to_points(depth_maps: Tensor, intrinsics: Tensor) -> Tensor:
    """Convert depth map(s) to pointcloud(s).

    Args:
        depth_map (Tensor): [B, H, W] or [H, W] depth values.
        intrinsics (Tensor): [B, 3, 3] or [3, 3] intrinsic matrix.

    Returns:
        Tensor: [B, H*W, 3] or [H*W, 3] 3D points.
    """
    squeeze = False
    if len(depth_maps.shape) == 2:
        depth_maps = depth_maps.unsqueeze(0)
        squeeze = True
    batch_size, height, width = depth_maps.shape
    points2d = create_meshgrid(height, width, normalized_coordinates=False, device=depth_maps.device)
    points2d = points2d.view(1, -1, 2).repeat(batch_size, 1, 1)
    points_ref = unproject_points(points2d, depth_maps.view(batch_size, -1), intrinsics)
    if squeeze:
        points_ref = points_ref.squeeze(0)
    return points_ref


def inverse_rigid_transform(transformation: Tensor) -> Tensor:
    """Calculate inverse of rigid body transformation(s).

    Args:
        transformation (Tensor): [..., 4, 4] transformations or single [4, 4]
            transformation.

    Returns:
        Tensor: Inverse of input transformation(s).
    """
    squeeze = False
    if len(transformation.shape) == 2:
        transformation = transformation.unsqueeze(0)
        squeeze = True
    rotation, translation = transformation[..., :3, :3], transformation[..., :3, 3]
    rot = rotation.transpose(-2, -1)
    t = -rot @ translation[..., None]
    inv = torch.cat([torch.cat([rot, t], -1), transformation[..., 3:4, :]], 1)
    if squeeze:
        inv = inv.squeeze(0)
    return inv


def rotate_z(pts: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate points around the z-axis.

    Args:
        pts: Tensor of shape (..., 3).
        angle: Rotation angle in radians.

    Returns:
        Tensor of shape (..., 3).
    """
    c, s = torch.cos(angle), torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    R = torch.stack([c, -s, zeros, s, c, zeros, zeros, zeros, ones]).view(3, 3)
    return torch.einsum("...ij,...j->...i", R, pts)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


@torch.jit.script
def angle_lerp(start: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    """Linear interpolation of two angles.

    Args:
        start: Starting angle.
        end: Ending angle.
        weight: Interpolation weight.

    Returns:
        Interpolated angle.
    """
    shortest_angle = ((((end - start) % (2 * torch.pi)) + (3 * torch.pi)) % (2 * torch.pi)) - torch.pi
    return start + shortest_angle * weight
