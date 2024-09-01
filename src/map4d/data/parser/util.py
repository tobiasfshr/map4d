import math
from pathlib import Path
from typing import Literal

import numpy as np
import pyquaternion
import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from torch import Tensor

from map4d.common.geometry import opencv_to_opengl, points_inside_image, project_points
from map4d.common.pointcloud import transform_points
from map4d.data.parser.typing import AnnotationInfo, ImageInfo


def orient_align_scale(
    images: list[ImageInfo],
    annotations: list[AnnotationInfo],
    bounds: np.ndarray,
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
    orient_center_poses: bool = True,
    auto_scale_poses: bool = True,
    use_depth_bounds: bool = True,
) -> tuple[np.ndarray, int]:
    """Orient, align and scale the poses, annotations, scene / object points and bounds. All ops are in-place."""
    # compute scale factor, transformation, apply to poses
    poses = torch.from_numpy(np.array([im.pose for im in images])).float()
    # orient and center poses
    transform_matrix = torch.eye(4)
    if orient_center_poses:
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=center_method,
        )

    scale_factor = 1.0
    if auto_scale_poses:
        if use_depth_bounds:
            scale_factor /= (bounds[3:] - bounds[:3]).max() / 2.0  # scale to [-1, 1]
        else:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))

    poses[:, :3, 3] *= scale_factor

    for im, pose in zip(images, poses):
        im.pose = pose
        if im.cam2vehicle is not None:
            im.cam2vehicle[:3, 3] *= scale_factor

    # Scale annotations
    for ann in annotations:
        if ann.vehicle2world is not None:
            ann.vehicle2world = to4x4(transform_matrix) @ ann.vehicle2world
            ann.vehicle2world[:3, 3] *= scale_factor

        if len(ann.boxes) == 0:
            continue

        ann.boxes[:, :3] = (transform_matrix[None, :3, :3] @ ann.boxes[:, :3, None]).squeeze(-1) + transform_matrix[
            :3, 3
        ]
        ann.boxes[:, :6] *= scale_factor

        for i, box in enumerate(ann.boxes):
            rotation_matrix = np.array(
                [[np.cos(box[6]), -np.sin(box[6]), 0.0], [np.sin(box[6]), np.cos(box[6]), 0.0], [0.0, 0.0, 1.0]]
            )
            rotation_matrix = transform_matrix[:3, :3].numpy() @ rotation_matrix
            yaw = pyquaternion.Quaternion(matrix=rotation_matrix, rtol=1e-04, atol=1e-06).yaw_pitch_roll[0]
            ann.boxes[i, 6] = float(yaw)

    # transform bounds if calculated
    bounds.shape = (2, 3)  # in-place reshape
    bounds @= transform_matrix[:3, :3].T.numpy()
    bounds += transform_matrix[:3, 3].numpy()
    bounds *= scale_factor

    return transform_matrix, scale_factor


def get_split_indices(train_split_fraction: float, images: list[ImageInfo], split: str = "train"):
    """Get train / eval splits. Aligned with SUDS splits.

    See https://github.com/hturki/suds/blob/9b472ad5e4fd4d810682af984ef65bfaf4d75188/scripts/metadata_utils.py#L135
    """
    sequence_ids = set([im.sequence_id for im in images])
    frame_ids = []
    for seq_id in sequence_ids:
        frame_ids.append(sorted(list(set([im.frame_id for im in images if im.sequence_id == seq_id]))))

    # filter images and poses based on train/eval split percentage
    num_snapshots = sum([len(frame_ids) for frame_ids in frame_ids])
    frames_all = np.arange(num_snapshots)
    if train_split_fraction == 1.0:
        frames_train = frames_all
        frames_eval = frames_all
    elif train_split_fraction > 0.5:
        test_every = math.floor(1 / (1.0 - train_split_fraction))
        frames_eval = np.arange(test_every, num_snapshots, test_every)
        frames_train = np.setdiff1d(frames_all, frames_eval)  # train images are the remaining images
    else:
        train_every = math.floor(1 / train_split_fraction)
        frames_train = np.arange(0, num_snapshots, train_every)
        frames_eval = np.setdiff1d(frames_all, frames_train)  # eval images are the remaining images
    assert (
        abs(len(frames_train) - math.ceil(train_split_fraction * num_snapshots)) <= 1
    ), f"Train split is not correct: {len(frames_train)} vs {math.ceil(train_split_fraction * num_snapshots)}"

    def get_indices(frames):
        indices = []
        for idx in frames:
            seq_id = 0
            while idx >= len(frame_ids[seq_id]):
                idx -= len(frame_ids[seq_id])
                seq_id += 1

            frame_id = frame_ids[seq_id][idx]
            for i, im in enumerate(images):
                if im.sequence_id == seq_id and im.frame_id == frame_id:
                    indices.append(i)
        return indices

    i_train = get_indices(frames_train)
    i_eval = get_indices(frames_eval)

    if split == "train":
        indices = i_train
    elif split in ["val", "test"]:
        indices = i_eval
    else:
        raise ValueError(f"Unknown dataparser split {split}")
    return indices


def get_mars_split_indices(train_split_fraction: float, images: list[ImageInfo], split: str = "train"):
    """Get split indices using MARS NVS-50 split method."""
    sequence_ids = set([im.sequence_id for im in images])
    assert len(sequence_ids) == 1, "MARS split is only supported for single sequence datasets."
    assert train_split_fraction == 0.5, "MARS split is only supported for 50% NVS split."
    frame_ids = sorted(list(set([im.frame_id for im in images])))

    if split == "train":
        frame_ids = np.array([idx for i, idx in enumerate(frame_ids) if (i + 1) % 3 in [1, 2]])
    else:
        frame_ids = np.array([idx for i, idx in enumerate(frame_ids) if (i + 1) % 3 == 0])

    indices = []
    for frame_id in frame_ids:
        for i, im in enumerate(images):
            if im.frame_id == frame_id:
                indices.append(i)

    return indices


def get_split_data(
    train_split_fraction: float,
    images: list[ImageInfo],
    annotations: list[AnnotationInfo],
    split: str = "train",
    remove_unseen_objects: bool = True,
    remove_nonrigid: bool = False,
    mars_split: bool = False,
):
    if mars_split:
        indices = get_mars_split_indices(train_split_fraction, images, split)
    else:
        indices = get_split_indices(train_split_fraction, images, split)
    images = [images[i] for i in indices]

    # check if all objects are in train split
    all_object_ids = set(torch.tensor(np.concatenate([ann.box_ids for ann in annotations])).unique().numpy().tolist())

    train_seq_frame_ids = set([(im.sequence_id, im.frame_id) for im in images])

    def is_in_train(x):
        x = (x.sequence_id, x.frame_id)
        if split == "train":
            return True if x in train_seq_frame_ids else False
        else:
            return False if x in train_seq_frame_ids else True

    keep_object_ids = set(
        torch.from_numpy(np.concatenate([ann.box_ids for ann in annotations if is_in_train(ann)]))
        .unique()
        .numpy()
        .tolist()
    )

    if not remove_unseen_objects:
        assert (
            keep_object_ids == all_object_ids
        ), f"All objects must be in train split, {all_object_ids - keep_object_ids} are missing."

    if remove_unseen_objects or remove_nonrigid:
        objs_to_remove = set()
        if remove_unseen_objects and keep_object_ids != all_object_ids:
            objs_to_remove = objs_to_remove.union(all_object_ids - keep_object_ids)
            CONSOLE.log(
                f"WARNING: Removing {len(objs_to_remove)} objects from dataset because they are unseen during training. They will be treated as static scene."
            )
        if remove_nonrigid:
            CONSOLE.log("Removing non-rigid objects from dataset.")
            all_pedestrian_ids = set(
                torch.from_numpy(np.concatenate([ann.box_ids[ann.class_ids == 1] for ann in annotations]))
                .unique()
                .numpy()
                .tolist()
            )
            objs_to_remove = objs_to_remove.union(all_pedestrian_ids)
            keep_object_ids -= all_pedestrian_ids

        for ann in annotations:
            mask = ~np.isin(ann.box_ids, list(objs_to_remove))
            ann.boxes = ann.boxes[mask]
            ann.box_ids = ann.box_ids[mask]
            ann.class_ids = ann.class_ids[mask]

        if len(objs_to_remove) > 0:
            CONSOLE.log(
                f"Removed {len(objs_to_remove)} objects from dataset, new number of objects: {len(keep_object_ids)}."
            )
            CONSOLE.log("Rescaling object ids to be consecutive.")
            for ann in annotations:
                box_ids = ann.box_ids
                for i, box_id in enumerate(box_ids):
                    ann.box_ids[i] = list(keep_object_ids).index(box_id)

    return images, annotations, indices, list(keep_object_ids)


def get_uniform_points_on_sphere_fibonacci(
    num_points: int, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
) -> Tensor:
    # https://arxiv.org/pdf/0912.4540.pdf
    # Golden angle in radians
    phi = math.pi * (3.0 - math.sqrt(5.0))
    N = (num_points - 1) / 2
    i = torch.linspace(-N, N, num_points, device=device, dtype=dtype)
    lat = torch.arcsin(2.0 * i / (2 * N + 1))
    lon = phi * i

    # Spherical to cartesian
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], -1)


def generate_background_points(
    points: Tensor,
    aabb: Tensor,
    images: list[Path],
    w2cs: list[Tensor],
    intrinsics: list[Tensor],
    im_wh: list[tuple[int, int]],
    num_spheres: int = 1,
    num_points_per_sphere: int = 200000,
    generate_colors: bool = True,
    window_size: float = 0.005,
    device: torch.device = torch.device("cpu"),
    sphere_radii: list[float] = None,
):
    """Generate background points on increasing radius spheres around the scene AABB.

    Optionally uses GPU acceleration for point visibility checks and color retrieval.
    """
    points = points.to(device)
    aabb = aabb.to(device)
    opengl2opencv = torch.from_numpy(opencv_to_opengl).to(device)
    w2cs = [to4x4(w2c) if w2c.shape == (3, 4) else w2c for w2c in w2cs]
    w2cs = [opengl2opencv @ w2c.to(device) for w2c in w2cs]
    intrinsics = [intr.to(device) for intr in intrinsics]

    dtype = aabb.dtype
    bg_points, bg_cols = [], []
    scene_radius = (aabb[1] - aabb[0]).norm() / 2.0
    scene_center = (aabb[1] + aabb[0]) / 2
    for i in range(num_spheres):
        if sphere_radii is not None:
            r = scene_radius * sphere_radii[i]
        else:
            # exponential increase in sphere radius
            r = scene_radius ** (2 ** (i + 1))

        # We use 2*num_points_per_sphere to get the upper hemisphere
        sphere_points = r * get_uniform_points_on_sphere_fibonacci(
            2 * num_points_per_sphere, device=device, dtype=dtype
        )
        # Take upper hemisphere as in the previous implementation
        sphere_points = sphere_points[sphere_points[:, 2] > 0]
        sphere_points = sphere_points + scene_center.unsqueeze(0)

        # check which sphere points fall into the view frustum of any camera and are projected far enough from foreground
        visible_mask = torch.zeros(sphere_points.shape[0], dtype=torch.bool, device=device)
        sphere_colors = torch.zeros((sphere_points.shape[0], 3), dtype=torch.uint8, device=device)
        for w2c, intr, wh, im_path in zip(w2cs, intrinsics, im_wh, images):
            if visible_mask.all():
                break

            width, height = wh
            window_pixel_size = max(width, height) * window_size
            downscale_factor = window_pixel_size // 2
            intr = intr.clone()
            intr[:2] /= downscale_factor
            height, width = height // downscale_factor, width // downscale_factor
            projected_mask = torch.zeros((int(height), int(width)), dtype=torch.bool, device=device)

            # project foreground and create an occupancy mask
            cam_front_points = transform_points(points, w2c)
            pix_front_points = project_points(cam_front_points, intr)
            inside_mask = points_inside_image(pix_front_points, cam_front_points[:, 2], (height, width))
            pix_front_points = pix_front_points[inside_mask]
            projected_mask[
                torch.round(pix_front_points[:, 1]).long(), torch.round(pix_front_points[:, 0]).long()
            ] = True

            # project background
            cam_points = transform_points(sphere_points[~visible_mask], w2c)
            pix_points = project_points(cam_points, intr)

            # check which points are visible
            inside_mask = points_inside_image(pix_points, cam_points[:, 2], (height, width))
            empty_mask = ~projected_mask[
                torch.round(pix_points[inside_mask][:, 1]).long(), torch.round(pix_points[inside_mask][:, 0]).long()
            ]
            full_mask = inside_mask.clone()
            full_mask[inside_mask] &= empty_mask

            # upscale to original resolution and retrieve colors for new points, if any
            new_points_mask = full_mask & ~visible_mask[~visible_mask]
            if new_points_mask.any() and generate_colors:
                pix_points = pix_points[new_points_mask]
                pix_points *= downscale_factor
                pix_points = pix_points.round().long()
                colors = torch.from_numpy(np.array(Image.open(im_path))).to(device)
                colors = colors[pix_points[:, 1], pix_points[:, 0]]
                color_mask = ~visible_mask.clone()
                color_mask[~visible_mask] = new_points_mask
                sphere_colors[color_mask] = colors

            # mask points that are visible and projected far enough from foreground
            visible_mask[~visible_mask] |= full_mask

        sphere_points = sphere_points[visible_mask]
        sphere_colors = sphere_colors[visible_mask]
        bg_points.append(sphere_points)
        bg_cols.append(sphere_colors)
    return torch.cat(bg_points), torch.cat(bg_cols)


def average_object_dimensions(annotations: list[AnnotationInfo]) -> None:
    """Average the object dimensions for each object id (in-place)."""
    avg_dims = {}
    for ann in annotations:
        for box, box_id in zip(ann.boxes, ann.box_ids):
            if box_id not in avg_dims:
                avg_dims[box_id] = box[3:6][None, :]
            else:
                avg_dims[box_id] = np.concatenate([avg_dims[box_id], box[3:6][None, :]], axis=0)

    for ann in annotations:
        for i, box_id in enumerate(ann.box_ids):
            ann.boxes[i, 3:6] = avg_dims[box_id].mean(axis=0)
