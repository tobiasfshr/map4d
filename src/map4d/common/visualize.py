import colorsys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import colormaps
from nerfstudio.utils.poses import to4x4
from PIL import Image, ImageDraw
from torch import Tensor

from map4d.common.geometry import inverse_rigid_transform, opencv_to_opengl, points_inside_image, project_points


def generate_colors():
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [
        15,
        13,
        25,
        12,
        19,
        8,
        22,
        24,
        29,
        17,
        28,
        20,
        2,
        27,
        11,
        26,
        21,
        4,
        3,
        18,
        9,
        5,
        14,
        1,
        16,
        0,
        23,
        7,
        6,
        10,
    ]
    colors = [tuple(map(lambda x: x * 255, colors[idx])) for idx in perm]
    return colors


COLORS = generate_colors()


def apply_depth_colormap(
    depth: torch.Tensor, min_d=None, max_d=None, colormap_options=colormaps.ColormapOptions("inferno_r")
):
    """Apply a colormap to a depth image

    Args:
        depth (torch.Tensor): depth image
        min_d (float, optional): minimum depth value.
        max_d (float, optional): maximum depth value.
        colormap_options (colormaps.ColormapOptions, optional): colormap options. Defaults to colormaps.ColormapOptions("inferno_r").

    Returns:
        torch.Tensor: color coded image
    """
    if min_d is None:
        min_d = depth.min()
    if max_d is None:
        max_d = depth.max()

    depth = (depth - min_d) / (max_d - min_d + 1e-10)
    depth[depth < 0] = 0
    depth[depth > 1] = torch.inf
    colored_image = colormaps.apply_colormap(depth, colormap_options=colormap_options)
    return colored_image


def get_bbox_points_dense(corners_per_bbox: np.ndarray, box_ids: np.ndarray, scene_scale: float):
    bbox_points, bbox_colors = [], []
    combinations = [
        (0, 1),
        (0, 4),
        (1, 5),
        (4, 5),
        (2, 3),
        (6, 7),
        (2, 6),
        (3, 7),
        (0, 3),
        (1, 2),
        (5, 6),
        (4, 7),
        (0, 5),
        (1, 4),
    ]
    if box_ids is None:
        box_ids = [None] * len(corners_per_bbox)
    for corners, box_id in zip(corners_per_bbox, box_ids):
        curr_bbox = corners.tolist()
        for p1, p2 in combinations:
            diff = corners[p1] - corners[p2]
            steps_cm = abs(int(np.linalg.norm(diff, 2) / scene_scale * 100))
            for i in range(1, steps_cm):
                curr_bbox.append(corners[p2] + (diff / steps_cm) * i)
        if box_id is not None:
            box_col = COLORS[int(box_id) % len(COLORS)]
        else:
            box_col = (255, 0, 0)
        bbox_colors.extend([box_col] * len(curr_bbox))
        bbox_points.extend(curr_bbox)
    return np.asarray(bbox_points), np.asarray(bbox_colors)


def boxes3d_to_corners3d(object_poses: Tensor, object_dimensions: Tensor) -> Tensor:
    """Get the 3D corners of object 3D bounding boxes.

    Args:
        object_poses (Tensor): N, 4
        object_dimensions (Tensor): N, 3 (wlh)

    Returns:
        Tensor: N, 8, 3 corners.

               (back)
        (3) +---------+. (2)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (4) ` +---------+ (5)
                     (front)
    """
    # corners in object space
    corners = torch.tensor(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [1, -1, -1],
        ],
        device=object_poses.device,
        dtype=object_poses.dtype,
    )
    corners = corners * object_dimensions[:, None, :] / 2
    # z axis angle to rotation matrix
    rotation_matrix = torch.stack(
        [
            torch.stack(
                [torch.cos(object_poses[:, 3]), -torch.sin(object_poses[:, 3]), torch.zeros_like(object_poses[:, 3])],
                dim=1,
            ),
            torch.stack(
                [torch.sin(object_poses[:, 3]), torch.cos(object_poses[:, 3]), torch.zeros_like(object_poses[:, 3])],
                dim=1,
            ),
            torch.tensor([0.0, 0.0, 1.0], device=object_poses.device).repeat(object_poses.shape[0], 1),
        ],
        dim=1,
    )
    # transform to world space
    corners = corners @ rotation_matrix.transpose(1, 2) + object_poses[:, None, :3]
    return corners


@dataclass
class BEVCanvas:
    """BEV canvas"""

    image: ImageDraw.Draw
    aabb: np.ndarray  # 2, 2
    scale: int

    def numpy(self) -> np.ndarray:
        return np.array(self.image._image, dtype=np.uint8)

    def tensor(self) -> Tensor:
        return torch.from_numpy(self.numpy())


def get_canvas_bev(
    aabb: Tensor,
    scale: int = 1000,
) -> BEVCanvas:
    """Create a blank canvas in BEV space."""
    bev_aabb = aabb[:, :2].cpu().numpy()
    bev_range = bev_aabb[1] - bev_aabb[0]

    # Generate figure size
    figure_hw = (int(bev_range[1] * scale), int(bev_range[0] * scale))
    white_image = np.ones([*figure_hw, 3]) * 255
    image = white_image.astype(np.uint8)
    image_draw = ImageDraw.Draw(Image.fromarray(image))

    return BEVCanvas(image_draw, bev_aabb, scale)


def draw_points_bev(
    canvas: BEVCanvas,
    points: Tensor,
    radius: int = 2,
    color: Optional[tuple[int, int, int]] = None,
    min_brightness: int = 25,
    max_brightness: int = 230,
) -> BEVCanvas:
    """Draw points in the BEV.

    Args:
        canvas (BEVCanvas): BEV canvas
        points (Tensor): N, 3
        radius (int, optional): Point radius. Defaults to 2.
        color (Optional[tuple[int, int, int]], optional): Point color. Defaults to None.
        min_brightness (int, optional): Minimum brightness of points based on distance. Defaults to 25.
        max_brightness (int, optional): Maximum brightness of points based on distance. Defaults to 230.

    Returns:
        BEVCanvas: BEV canvas with points drawn.

    """
    bev_aabb = canvas.aabb
    scale = canvas.scale
    figure_hw = canvas.image._image.size[::-1]

    # get seed points pixel coordinates
    points = points[:, :2].detach().cpu().numpy()
    points = ((points - bev_aabb[0]) * scale).astype(np.int64)
    # filter out of image seed points
    points = points[
        (points[:, 0] >= 0) & (points[:, 0] < figure_hw[1]) & (points[:, 1] >= 0) & (points[:, 1] < figure_hw[0])
    ]

    # draw seed points
    def draw_circle(center, color):
        x1 = center[0] - radius
        y1 = center[1] - radius
        x2 = center[0] + radius
        y2 = center[1] + radius
        canvas.image.ellipse((x1, y1, x2, y2), fill=color, outline=color)

    diag = np.linalg.norm(np.array(figure_hw) / 2)
    for point in points:
        if color is None:
            grey_level = int(np.linalg.norm(point - np.array(figure_hw) / 2) / diag * max_brightness) + min_brightness
            color = (grey_level, grey_level, grey_level)
        draw_circle(point, color)

    return canvas


def draw_boxes_bev(
    canvas: BEVCanvas,
    object_poses: Tensor,
    object_dimensions: Tensor,
    object_ids: Optional[Tensor] = None,
    line_width: int = 3,
    color: Optional[tuple[int, int, int]] = None,
) -> BEVCanvas:
    """Draws N 3D boxes in the BEV. Slow, use for debugging only.

    Args:
        canvas (BEVCanvas): BEV canvas
        object_poses (Tensor): N, 4
        object_dimensions (Tensor): N, 3
        object_ids (Tensor): N
        line_width (int, optional): line width. Defaults to 3.
        color (Optional[tuple[int, int, int]], optional): box color. Defaults to None.

    Returns:
        BEVCanvas: BEV canvas with boxes drawn.

    """
    bev_aabb = canvas.aabb
    scale = canvas.scale
    figure_hw = canvas.image._image.size[::-1]

    # get BEV corners pixel coordinates
    corners = boxes3d_to_corners3d(object_poses.detach(), object_dimensions)
    corners = corners[:, :4, :2].cpu().numpy()
    corners = ((corners - bev_aabb[0]) * scale).astype(np.int64)
    # filter out of image boxes
    corners = corners[
        (corners[:, :, 0] >= 0).all(axis=1)
        & (corners[:, :, 0] < figure_hw[1]).all(axis=1)
        & (corners[:, :, 1] >= 0).all(axis=1)
        & (corners[:, :, 1] < figure_hw[0]).all(axis=1)
    ]

    def draw_line(point1, point2, color):
        canvas.image.line((point1, point2), width=line_width, fill=color)

    for i, corners_box in enumerate(corners):
        if color is None:
            if object_ids is not None:
                obj_id = object_ids[i]
                color = COLORS[obj_id % len(COLORS)]
                color = tuple(map(int, color))
            else:
                color = (255, 0, 0)

        draw_line(tuple(corners_box[0]), tuple(corners_box[1]), color)
        draw_line(tuple(corners_box[1]), tuple(corners_box[2]), color)
        draw_line(tuple(corners_box[2]), tuple(corners_box[3]), color)
        draw_line(tuple(corners_box[3]), tuple(corners_box[0]), color)
        center_forward = np.mean(corners_box[:2], axis=0, dtype=np.float32).astype(np.int64)
        center = np.mean(corners_box, axis=0, dtype=np.float32).astype(np.int64)
        draw_line(tuple(center.tolist()), tuple(center_forward.tolist()), color)
    return canvas


@torch.no_grad()
def draw_boxes_in_image(
    image: Tensor,
    camera_pose: Tensor,
    camera_intrinsics: Tensor,
    object_poses: Tensor,
    object_dimensions: Tensor,
    object_ids: Tensor,
    linewidth: float = 2,
) -> Tensor:
    """Draws N 3D boxes in the image. Slow, use for debugging only.

    Args:
        image (Tensor): H, W, 3, float 0..1
        camera_pose (Tensor): N, 4, 4
        camera_intrinsics (Tensor): N, 3, 3
        object_poses (Tensor): N, 4
        object_dimensions (Tensor): N, 3
        object_ids (Tensor): N
        linewidth (float, optional): line width. Defaults to 2.

    Returns:
        Tensor: H, W, 3 image with boxes drawn.
    """
    width, height = image.shape[1], image.shape[0]
    corners = boxes3d_to_corners3d(object_poses, object_dimensions)
    # transform to camera space
    corners = corners @ camera_pose[:3, :3] - camera_pose[:3, 3] @ camera_pose[:3, :3]
    corners = corners @ torch.from_numpy(opencv_to_opengl[:3, :3]).to(corners)

    # project to image, filter visible
    depth = corners[..., 2]
    corners = project_points(corners, camera_intrinsics.to(corners.device))
    visible_mask = points_inside_image(corners, depth, (height, width)).any(dim=1)

    if not (visible_mask).any():
        return image

    corners = corners[visible_mask]

    image_np = image.cpu().numpy() * 255

    def draw_rect(selected_corners):
        prev = tuple(selected_corners[-1])
        for corner in selected_corners:
            cv2.line(image_np, prev, tuple(corner), color, linewidth)
            prev = tuple(corner)

    for corners_box, obj_id in zip(corners, object_ids[visible_mask]):
        corners_box = corners_box.cpu().numpy().astype(int)
        color = COLORS[obj_id % len(COLORS)]
        cv2.putText(
            image_np,
            f"ID: {obj_id}",
            (corners_box[4, 0], corners_box[4, 1] - 8),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color=color,
        )

        # Draw the sides
        for i in range(4):
            cv2.line(image_np, tuple(corners_box[i]), tuple(corners_box[i + 4]), color, linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners)
        draw_rect(corners_box[:4])
        draw_rect(corners_box[4:])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners_box[4:6], axis=0).astype(int)
        center_bottom = np.mean(corners_box[4:8], axis=0).astype(int)
        cv2.line(image_np, tuple(center_bottom), tuple(center_bottom_forward), color, linewidth)

    return torch.from_numpy(image_np).to(image) / 255


def rasterize_points(
    camera: Cameras,
    xyz: Tensor,
    colors: Tensor,
    znear: float = 1e-5,
    zfar: float = float("inf"),
    marker_size: int = 10,
    out=None,
):
    """Rasterize points in the camera view."""
    c2w = camera.camera_to_worlds.clone()[0]
    c2w = to4x4(c2w) @ torch.tensor(opencv_to_opengl, dtype=c2w.dtype, device=c2w.device)

    w2c = inverse_rigid_transform(c2w)
    points = xyz
    points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    assert len(w2c.shape) == 2
    points = points @ w2c.T
    points = points[..., :3] / points[..., 3:]
    depths = points[..., 2]
    valid_mask = torch.logical_and(znear < depths, depths < zfar)
    points = points[..., :2] / points[..., 2:3]

    # Currently points are in [0, 1] range
    # Undistort would apply here

    # Move to
    x, y = points.unbind(-1)
    x = x * camera.fx[0] + camera.cx[0] - 0.5
    y = y * camera.fy[0] + camera.cy[0] - 0.5
    points2D = torch.stack([x, y], -1)

    # Rasterize
    points2D = points2D[valid_mask].int().detach().cpu().numpy()
    colors = colors.view(-1, colors.shape[-1])[valid_mask].detach().cpu().numpy()

    # sort by depths
    order = (-depths[valid_mask]).argsort(0).cpu()
    points2D = points2D[order]
    colors = colors[order]

    if out is None:
        img = np.ndarray((camera.height[0], camera.width[0], 3), dtype=np.uint8)
        img.fill(255)
    else:
        img = out.cpu().numpy()
        if img.dtype != np.uint8:
            img = (img * 255.0).astype(np.uint8)
    for pt, color in zip(points2D, colors):
        if len(color) == 3 or color[3] > 0.5:
            cv2.circle(img, pt.tolist(), marker_size, color[:3].tolist(), thickness=cv2.FILLED)
    if out is not None:
        if out.dtype == torch.uint8:
            out.copy_(torch.from_numpy(img))
        else:
            out.copy_(torch.from_numpy(img).float() / 255.0)
        return out
    return torch.from_numpy(img)
