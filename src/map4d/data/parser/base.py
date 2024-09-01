"""Data parser for street scene datasets."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from map4d.common.geometry import inverse_rigid_transform
from map4d.common.io import from_ply
from map4d.common.pointcloud import get_points_in_boxes3d, voxelize_pointcloud
from map4d.data.parser.util import (
    average_object_dimensions,
    generate_background_points,
    get_split_data,
    orient_align_scale,
)


@dataclass
class StreetSceneParserConfig(DataParserConfig):
    """Basic street scene dataset parser config"""

    _target: Type = field(default_factory=lambda: StreetSceneParser)
    """target class to instantiate"""
    orient_center_poses: bool = True
    """Whether to automatically orient and center the poses."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    use_depth_bounds: bool = True
    """Whether to compute the scene bounds from the depth images."""
    remove_unseen_objects: bool = True
    """Whether to remove objects that are not seen during training."""
    enlarge_boxes: float = 0.25
    """Enlarge each box at each dim (in meters)."""
    load_pointclouds: bool = False
    """Whether to load the pointclouds of the scene and objects."""
    voxel_size: float = 0.1
    """Voxel size for pointclouds. If 0, no voxelization is performed."""
    remove_nonrigid: bool = True
    """Whether to remove pedestrian annotations or not."""
    add_background_points: bool = False
    """Whether to add background points to the scene pointcloud to cover sky / far away regions."""
    num_background_spheres: int = 3
    """Number of spheres the background points lie on."""
    sphere_radii: Optional[list[float]] = None
    """Radii of the spheres the background points lie on (e.g. 1.0 = 1.0 * scene radius)."""
    points_per_sphere: int = 100000
    """Number of points per sphere."""
    load_background_colors: bool = False
    """Whether to load the colors for the background points from the images or not."""
    sequence_ids: Optional[list[int]] = None
    """Sequence ids to load. If None, all sequences are loaded."""
    mars_split: bool = False
    """Whether to use the MARS NVS-50 split method for the dataset.
    https://github.com/OPEN-AIR-SUN/mars/blob/563138c092f320c3f8c1e4d13be29ec80dc2ca81/mars/data/mars_kitti_dataparser.py#L1097
    """


class StreetSceneParser(DataParser):
    """Basic dataparser for street scene datasets.

    We provide the data in the following order: The data should be sorted by:
        1. Sequences
        2. Frames
        3. Cameras
    This is necessary for the pose optimization, which assumes a certain ordering of the poses
    to group the pose deltas for all cameras in a frame together.

    NOTE: Object poses are independent from train / eval split, thus provided as a single list.
    """

    config: StreetSceneParserConfig

    def __init__(self, config: StreetSceneParserConfig):
        super().__init__(config)
        self._cache = None

    def _load_info(self):
        # load street scene representation
        assert os.path.exists(self.config.data), f"Metadata file {self.config.data} does not exist."
        with open(self.config.data, "rb") as f:
            images, annotations, bounds, pointclouds, depth_unit_scale_factor = pickle.load(f)

        # [verification] average object dimensions of boxes belonging to the same box id across all frames
        average_object_dimensions(annotations)

        if self.config.enlarge_boxes > 0:
            for ann in annotations:
                ann.boxes[:, 3:6] += self.config.enlarge_boxes

        if self.config.sequence_ids is not None:
            images = [im for im in images if im.sequence_id in self.config.sequence_ids]
            annotations = [ann for ann in annotations if ann.sequence_id in self.config.sequence_ids]
            pointclouds = [pc for pc in pointclouds if pc.sequence_id in self.config.sequence_ids]

            # rescale object ids to be contiguous
            object_ids = set(
                torch.from_numpy(np.concatenate([ann.box_ids for ann in annotations])).unique().numpy().tolist()
            )
            for ann in annotations:
                box_ids = ann.box_ids
                for i, box_id in enumerate(box_ids):
                    ann.box_ids[i] = list(object_ids).index(box_id)

            # rescale sequence ids to be contiguous
            sequence_ids = set(
                torch.from_numpy(np.array([ann.sequence_id for ann in annotations])).unique().numpy().tolist()
            )
            for ann in annotations:
                ann.sequence_id = list(sequence_ids).index(ann.sequence_id)
            for im in images:
                im.sequence_id = list(sequence_ids).index(im.sequence_id)
            for pc in pointclouds:
                pc.sequence_id = list(sequence_ids).index(pc.sequence_id)

        # timestamp / frame id normalization to approx. -1; 1 range
        # assumes timestamp starting from same ref time (e.g. 0) at each seq
        timestamps = [im.timestamp if im.timestamp is not None else im.frame_id for im in images]
        max_time, min_time = max(timestamps), min(timestamps)
        for im in images:
            im.timestamp = (im.timestamp if im.timestamp is not None else im.frame_id - min_time) / (
                max_time - min_time
            ) * 2.0 - 1.0
            assert -1.2 <= im.timestamp <= 1.2, f"Timestamps must be normalized, got {im.timestamp}."

        for ann in annotations:
            ann.timestamp = (ann.timestamp if ann.timestamp is not None else ann.frame_id - min_time) / (
                max_time - min_time
            ) * 2.0 - 1.0
            assert -1.2 <= ann.timestamp <= 1.2, f"Timestamps must be normalized, got {ann.timestamp}."

        return images, annotations, bounds, pointclouds, depth_unit_scale_factor

    def load_points(
        self, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Split background and foreground pointclouds with annotations, voxelize."""
        assert self._cache is not None, "Cache must be loaded before loading points."
        images, annotations, bounds, pointclouds, transform_matrix, scale_factor, _ = self._cache
        transform_matrix = transform_matrix.to(device)
        train_seq_frame_ids = set(
            [(im.sequence_id, im.frame_id) for i, im in enumerate(images) if i in self.train_image_indices]
        )
        train_indices = [i for i, pc in enumerate(pointclouds) if (pc.sequence_id, pc.frame_id) in train_seq_frame_ids]

        object_points = {}
        object_colors = {}
        scene_points = []
        scene_colors = []
        CONSOLE.log("Loading and processing pointclouds...")
        for i, (pc, ann) in tqdm(enumerate(zip(pointclouds, annotations))):
            assert (
                pc.sequence_id == ann.sequence_id and pc.frame_id == ann.frame_id
            ), "Pointcloud and Annotation list are assumed be in corresponding order."

            if i not in train_indices:
                continue

            pc, colors = from_ply(pc.points_path)
            if colors is None:
                colors = np.zeros_like(pc)
            colors = torch.from_numpy(colors).float().to(device)
            pc = torch.from_numpy(pc).float().to(device)
            # transform pointclouds
            pc @= transform_matrix[:3, :3].T
            pc += transform_matrix[:3, 3]

            boxes = torch.from_numpy(ann.boxes).float().to(device)
            boxes[:, :6] /= scale_factor
            indices, obj_points = get_points_in_boxes3d(pc, boxes)
            mask = indices == -1
            for i, (obj_id, obj_pc) in enumerate(zip(ann.box_ids, obj_points)):
                assert (
                    obj_pc.shape[0] == colors[indices == i].shape[0]
                ), f"Pointcloud and colors must have same length: {obj_pc.shape[0]} != {colors[indices == i].shape[0]}"
                if obj_id not in object_points:
                    object_points[obj_id] = obj_pc
                    object_colors[obj_id] = colors[indices == i]
                else:
                    object_points[obj_id] = torch.cat([object_points[obj_id], obj_pc])
                    object_colors[obj_id] = torch.cat([object_colors[obj_id], colors[indices == i]])
            scene_points.append(pc[mask])
            scene_colors.append(colors[mask])

        # concat, voxelize scene and object points
        scene_points = torch.cat(scene_points)
        scene_colors = torch.cat(scene_colors)
        scene_points, scene_colors = voxelize_pointcloud(scene_points, scene_colors, self.config.voxel_size)

        object_points_list = [None] * len(object_points)
        object_colors_list = [None] * len(object_points)
        for obj, pc, col in zip(object_points.keys(), object_points.values(), object_colors.values()):
            object_points_list[obj], object_colors_list[obj] = voxelize_pointcloud(
                pc, col, voxel_size=self.config.voxel_size
            )
        object_points = object_points_list
        object_colors = object_colors_list
        assert all(
            [obj is not None for obj in object_points]
        ), "Object points must be provided for all objects and ids must be contiguous starting from 0."

        # scale scene / object points
        scene_points *= scale_factor
        for obj_pts in object_points:
            obj_pts *= scale_factor

        assert not torch.isnan(scene_points).any(), "Scene points must not contain NaN values."
        assert not any(
            torch.isnan(obj_pts).any() for obj_pts in object_points
        ), "Object points must not contain NaN values."

        CONSOLE.log(
            f"Pointclouds loaded and processed: {scene_points.shape[0]} static points, {sum((o.shape[0] for o in object_points))} dynamic points."
        )
        # add background points
        if self.config.add_background_points:
            CONSOLE.log("Generating background points...")
            bg_points, bg_colors = generate_background_points(
                scene_points,
                torch.from_numpy(bounds).float(),
                [Path(im.image_path) for im in images],
                [inverse_rigid_transform(im.pose) for im in images],
                [torch.from_numpy(im.intrinsics).float() for im in images],
                [(im.width, im.height) for im in images],
                self.config.num_background_spheres,
                self.config.points_per_sphere,
                self.config.load_background_colors,
                device=device,
                sphere_radii=self.config.sphere_radii,
            )
            scene_points = torch.cat([scene_points, bg_points])
            scene_colors = torch.cat([scene_colors, bg_colors])
            CONSOLE.log(f"Added {bg_points.shape[0]} background points to the scene.")

        return scene_points, scene_colors, object_points, object_colors

    def _generate_dataparser_outputs(self, split: str = "train"):
        if self._cache is None:
            # processing before data split
            images, annotations, bounds, pointclouds, depth_unit_scale_factor = self._load_info()

            # compute scaling and transformation, modify info in-place
            transform_matrix, scale_factor = orient_align_scale(
                images,
                annotations,
                bounds,
                self.config.orientation_method,
                self.config.center_method,
                self.config.orient_center_poses,
                self.config.auto_scale_poses,
                self.config.use_depth_bounds,
            )

            # MARS trains on frame [0, 1] and evaluates on frame [3], discarding frame 2
            if self.config.mars_split:
                frame_ids = sorted(list(set([im.frame_id for im in images])))
                frame_ids = [idx for i, idx in enumerate(frame_ids) if (i + 1) % 4 in [1, 2, 0]]
                images = [im for im in images if im.frame_id in frame_ids]
                annotations = [ann for ann in annotations if ann.frame_id in frame_ids]
                pointclouds = [pc for pc in pointclouds if pc.frame_id in frame_ids]

            # processing before data split done, save to cache
            self._cache = (
                images,
                annotations,
                bounds,
                pointclouds,
                transform_matrix,
                scale_factor,
                depth_unit_scale_factor,
            )
        else:
            (
                images,
                annotations,
                bounds,
                pointclouds,
                transform_matrix,
                scale_factor,
                depth_unit_scale_factor,
            ) = self._cache

        # Filter based on dataset split
        images, annotations, split_indices, keep_obj_ids = get_split_data(
            self.config.train_split_fraction,
            images,
            annotations,
            split,
            self.config.remove_unseen_objects,
            self.config.remove_nonrigid,
            self.config.mars_split,
        )
        # NOTE: The above function modifies the input list _elements_ in-place, so that load_points
        # will return a filtered list of object points and we don't need to filter with keep_obj_ids
        # The return argument is still a new sublist, which is why we reassign images / annotations

        if split == "train":
            self.train_image_indices = split_indices

        # Generate dataparser outputs
        # Cameras
        im_hws = torch.tensor([(im.height, im.width) for im in images]).int()
        intrinsics = torch.from_numpy(np.stack([im.intrinsics for im in images])).float()
        cam_metadata = {
            "cam_idx": torch.tensor(split_indices).unsqueeze(-1).long(),
            "camera_ids": torch.from_numpy(np.array([im.cam_id for im in images])).unsqueeze(-1).long(),
            "sequence_ids": torch.from_numpy(np.array([im.sequence_id for im in images])).unsqueeze(-1).long(),
        }
        num_cameras = torch.from_numpy(np.array([im.cam_id for im in images])).int().unsqueeze(-1).unique().numel()

        # create cameras
        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=im_hws[:, 0:1],
            width=im_hws[:, 1:2],
            camera_to_worlds=torch.from_numpy(np.stack([im.pose for im in images])).float()[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor([im.timestamp for im in images]).float().unsqueeze(-1),
            metadata=cam_metadata,
        )

        # create scene box
        if not self.config.use_depth_bounds:
            aabb_scale = 1.0
            bounds = np.array([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]])
        scene_box = SceneBox(aabb=torch.from_numpy(bounds).float())

        # create dataparser outputs
        data_metadata = {
            # depth
            "depth_filenames": [Path(im.depth_path) if im.depth_path is not None else None for im in images],
            "depth_unit_scale_factor": depth_unit_scale_factor,
            "dataparser_scale": scale_factor,  # add for model to visualize depth maps with correct scaling
            # objects
            "object_seq_ids": torch.from_numpy(np.array([ann.sequence_id for ann in annotations])).long(),
            "object_times": torch.from_numpy(np.array([ann.timestamp for ann in annotations])).float(),
            "object_poses": [torch.from_numpy(ann.boxes[:, [0, 1, 2, 6]]).float() for ann in annotations],
            "object_dims": [torch.from_numpy(ann.boxes[:, [3, 4, 5]]).float() for ann in annotations],
            "object_class_ids": [torch.from_numpy(ann.class_ids).long() for ann in annotations],
            "object_ids": [torch.from_numpy(ann.box_ids).long() for ann in annotations],
            "keep_object_ids": keep_obj_ids,
            # image metadata
            "sequence_ids": torch.from_numpy(np.array([im.sequence_id for im in images])).long(),
            "frame_ids": torch.from_numpy(np.array([im.timestamp for im in images])).long(),
            "num_cameras": num_cameras,
        }

        mask_filenames = [Path(im.mask_path) if im.mask_path is not None else None for im in images]
        if all(mask is None for mask in mask_filenames):
            mask_filenames = None
        dataparser_outputs = DataparserOutputs(
            image_filenames=[Path(im.image_path) for im in images],
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            dataparser_scale=scale_factor,
            metadata=data_metadata,
        )
        return dataparser_outputs


StreetSceneDataParserSpecification = DataParserSpecification(
    config=StreetSceneParserConfig(),
    description="Dataparser for dynamic urban scenes",
)
