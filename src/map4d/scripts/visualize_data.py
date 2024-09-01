"""Scripts for visualizing the input data."""
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import mediapy as media
import numpy as np
import torch
import tyro
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from typing_extensions import Annotated

from map4d.common.geometry import depth_to_points, generate_depth_map, opencv_to_opengl, rotate_z
from map4d.common.io import get_depth_image_from_path, to_ply
from map4d.common.pointcloud import transform_points
from map4d.common.visualize import COLORS, apply_depth_colormap, boxes3d_to_corners3d, draw_boxes_in_image
from map4d.data.parser.base import StreetSceneParser, StreetSceneParserConfig
from map4d.data.parser.typing import AnnotationInfo, ImageInfo
from map4d.model.util import get_objects_at_time
from map4d.scripts.util import grid_cameras, tile_cameras


@dataclass
class BaseVisualize:
    """Base visualizer class containing utilities to load dataparser outputs and alike."""

    dataparser_config: StreetSceneParserConfig = field(default_factory=StreetSceneParserConfig)
    """Path to metadata file containing the scene information."""

    def __post_init__(self):
        self.dataparser = StreetSceneParser(self.dataparser_config)

    def main(self):
        """Main function for visualizing the data."""
        raise NotImplementedError


@dataclass
class VisualizeScene(BaseVisualize):
    """Visualize the scene as pointcloud from given depth images, including 3D bounding boxes."""

    visualize_every: int = 1
    """Visualize every nth image."""
    output_file: Path = Path("scene.ply")
    """Output file to save the pointcloud to."""
    max_points_per_view: int = 10000
    """Maximum number of points to visualize per view."""

    def main(self):
        """Main function for visualizing the data."""
        self.dataparser.config.train_split_fraction = 1.0
        dpo = self.dataparser.get_dataparser_outputs(split="train")

        im_hws = torch.cat((dpo.cameras.image_height, dpo.cameras.image_width), -1)[:: self.visualize_every]
        depths = dpo.metadata["depth_filenames"][:: self.visualize_every]
        poses, intrinsics = (
            dpo.cameras.camera_to_worlds[:: self.visualize_every],
            dpo.cameras.get_intrinsics_matrices()[:: self.visualize_every],
        )
        boxes3d = torch.cat([torch.cat(dpo.metadata["object_poses"]), torch.cat(dpo.metadata["object_dims"])], dim=-1)[
            :, [0, 1, 2, 4, 5, 6, 3]
        ]
        boxes3d_ids = torch.cat(dpo.metadata["object_ids"])
        scale_factor = dpo.dataparser_scale

        # check pointcloud, boxes3d: depths / poses / intrinsics / scaling
        all_points = []
        for pose, intr, dep, (h, w) in zip(poses, intrinsics, depths, im_hws):
            depth = get_depth_image_from_path(
                filepath=dep, height=int(h), width=int(w), scale_factor=float(scale_factor)
            )
            points = depth_to_points(depth.squeeze(-1), intr)
            points = points[points[:, 2] > 0]
            points = points[torch.randperm(points.shape[0])[: self.max_points_per_view]]

            # opencv to opengl
            points = points @ opencv_to_opengl[:3, :3]
            points = transform_points(points, pose)
            all_points.append(points)

        colors = torch.ones((sum([len(p) for p in all_points]), 3))
        if boxes3d is not None:
            box_points = boxes3d_to_corners3d(boxes3d[:, [0, 1, 2, 6]], boxes3d[:, [3, 4, 5]])
            box_colors = torch.zeros_like(box_points)

            if boxes3d_ids is not None:
                colors_ids = torch.tensor([COLORS[i % len(COLORS)] for i in boxes3d_ids])
                box_colors[:, [0, 1, 4, 5], 0] = 255  # front red
                box_colors[:, [2, 3, 6, 7]] = colors_ids[:, None, :]
            else:
                box_colors[:, [0, 1, 4, 5], 0] = 255  # front red
                box_colors[:, [2, 3, 6, 7], 1] = 255

            box_points = box_points.view(-1, 3)
            box_colors = box_colors.view(-1, 3)
            colors = torch.cat([colors, box_colors], dim=0)
            all_points.append(box_points)

        all_points = torch.cat(all_points, dim=0)
        to_ply(self.output_file, all_points, colors=colors)
        CONSOLE.log(f"Saved scene to {self.output_file}.")


@dataclass
class VisualizePointcloud(BaseVisualize):
    """Visualize the computed global pointcloud(s)."""

    output_path: Path = Path("pointclouds")
    time: float = 0.0
    """Time to visualize the pointcloud at."""
    sequence_id: int = 0
    """Sequence id to visualize."""

    def main(self):
        """Main function for visualizing the data."""
        train_dpo = self.dataparser.get_dataparser_outputs(split="train")
        scene_points, scene_colors, object_points, object_colors = self.dataparser.load_points()
        object_points, object_colors = self.transform_object_points(object_points, object_colors, train_dpo)
        camera_points, camera_colors = self.generate_camera_points(train_dpo)
        scene_points = torch.cat([scene_points, object_points, camera_points], dim=0)
        scene_colors = torch.cat([scene_colors, object_colors, camera_colors], dim=0)
        os.makedirs(self.output_path, exist_ok=True)
        to_ply(self.output_path / "pointcloud.ply", scene_points, scene_colors)
        CONSOLE.log(f"Saved to {self.output_path / 'pointcloud.ply'}.")

    def transform_object_points(self, points, colors, dpo):
        object_poses = dpo.metadata["object_poses"]
        object_dims = dpo.metadata["object_dims"]
        object_ids = dpo.metadata["object_ids"]
        object_times = dpo.metadata["object_times"]
        object_sequence_ids = dpo.metadata["object_seq_ids"]

        # get closest time with matching sequence id
        time_diffs = (object_times - self.time).abs()
        time_diffs[object_sequence_ids != self.sequence_id] = float("inf")
        idx = time_diffs.argmin()
        object_poses = object_poses[idx]
        object_dims = object_dims[idx]
        object_ids = object_ids[idx]

        # transform object points
        all_object_points, all_object_colors = [], []
        box_corners = boxes3d_to_corners3d(object_poses, object_dims)
        box_colors = torch.zeros_like(box_corners)
        box_colors[:, [0, 1, 4, 5], 1] = 255  # front green
        box_colors[:, [2, 3, 6, 7], 0] = 255
        for pose, obj_id, corners, corner_colors in zip(object_poses, object_ids, box_corners, box_colors):
            object_points = points[obj_id]
            object_colors = colors[obj_id]
            object_points = rotate_z(object_points, pose[3]) + pose[:3]
            object_points = torch.cat([object_points, corners], dim=0)
            object_colors = torch.cat([object_colors, corner_colors], dim=0)
            all_object_points.append(object_points)
            all_object_colors.append(object_colors)

        return torch.cat(all_object_points, dim=0), torch.cat(all_object_colors, dim=0)

    def generate_camera_points(self, dpo):
        camera_locs = dpo.cameras.camera_to_worlds[:, :3, 3]
        camera_colors = torch.zeros_like(camera_locs)
        camera_colors[:, 2] = 255
        return camera_locs, camera_colors


@dataclass
class VisualizeView(BaseVisualize):
    """Visualize a certain input view and its (sparse) depth GT."""

    image_idx: int = -1
    """Index of the image to visualize."""
    split: Literal["train", "val", "test"] = "val"
    """Split to use for visualization."""
    output_path: Path = Path("renders")
    """Path to save the gt images to."""
    min_depth: float = 1.0
    """Minimum depth for the depth colormap."""
    max_depth: float = 82.5
    """Maximum depth for the depth colormap."""

    def main(self):
        """Main function for visualizing the data."""
        assert self.image_idx >= 0, "Please specify a valid image index."
        dpo = self.dataparser.get_dataparser_outputs(split=self.split)
        shutil.copy(dpo.image_filenames[self.image_idx], self.output_path / f"{self.image_idx:05d}_rgb_gt.jpg")
        # depth gt generation
        depth = get_depth_image_from_path(
            filepath=dpo.metadata["depth_filenames"][self.image_idx],
            height=int(dpo.cameras.height[self.image_idx]),
            width=int(dpo.cameras.width[self.image_idx]),
            scale_factor=dpo.dataparser_scale,
        )

        # compute min and max scaled depth for depth colormap (s.t. it can be aligned with the predicted depth)
        min_d, max_d = self.min_depth * dpo.dataparser_scale, self.max_depth * dpo.dataparser_scale

        points = depth_to_points(
            depth.unsqueeze(0).squeeze(-1), dpo.cameras.get_intrinsics_matrices()[self.image_idx].unsqueeze(0)
        ).squeeze(0)
        # down and upscale by factor of 10 in order to densify sparse lidar points
        dpo.cameras.rescale_output_resolution(1 / 10)
        depth = generate_depth_map(
            points,
            dpo.cameras.get_intrinsics_matrices()[self.image_idx],
            (dpo.cameras.height[self.image_idx], dpo.cameras.width[self.image_idx]),
        ).unsqueeze(-1)
        depth[depth == 0] = torch.inf
        depth_image = apply_depth_colormap(depth, min_d, max_d)
        media.write_image(self.output_path / f"{self.image_idx:05d}_depth_gt.jpg", depth_image.cpu().numpy())
        dpo.cameras.rescale_output_resolution(10)


@dataclass
class VisualizeVideo(BaseVisualize):
    """Make video with all camera streams for a single sequence."""

    output_path: Path = Path("videos")
    """Output path to save the videos to."""
    sequence_id: int = 0
    """The sequence id to visualize."""
    camera_ids: int | tuple[int, ...] | None = None
    """The camera id(s) to visualize. Use None to visualize all cameras."""
    fps: int = 10
    """Frames per second for the output video."""
    cameras: tuple[str, ...] | None = (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    )
    """The camera names for tile_cameras. Use None to align cameras in a grid."""
    cameras_per_row: int = 4
    """Number of cameras per row for the grid layout."""

    def main(self):
        """Main function for visualizing the data."""
        self.dataparser.config.train_split_fraction = 1.0
        dpo = self.dataparser.get_dataparser_outputs(split="train")
        images, annotations, _, _, _, _, _ = self.dataparser._cache
        cameras = dpo.cameras

        # extract annotations
        obj_poses = dpo.metadata["object_poses"]
        obj_dims = dpo.metadata["object_dims"]
        obj_ids = dpo.metadata["object_ids"]
        obj_seq_ids = dpo.metadata["object_seq_ids"]
        obj_times = dpo.metadata["object_times"]

        def _to_tensor(x, pad_val=0):
            max_shape = torch.tensor([y.shape for y in x]).max(dim=0)[0]
            x_tensor = torch.ones((len(x), *max_shape)) * pad_val
            for i, y in enumerate(x):
                x_tensor[i, : len(y)] = y
            return x_tensor

        obj_ids = _to_tensor(obj_ids, pad_val=-1)
        obj_poses = _to_tensor(obj_poses)
        obj_dims = _to_tensor(obj_dims)

        os.makedirs(self.output_path, exist_ok=True)
        frames = sorted(list(set([im.frame_id for im in images if im.sequence_id == self.sequence_id])))
        output_frames = []
        cam_ids = self.camera_ids if self.camera_ids is not None else set([im.cam_id for im in images])
        if isinstance(cam_ids, int):
            cam_ids = [cam_ids]

        def load_image(im: ImageInfo, camera, ann: AnnotationInfo) -> np.ndarray:
            """Load image from path."""
            img = torch.from_numpy(np.array(Image.open(im.image_path))).float() / 255
            time = camera.times[0]
            sequence = camera.metadata["sequence_ids"][0]
            ids, poses, dims, _ = get_objects_at_time(
                obj_poses, obj_ids, obj_dims, obj_times, obj_seq_ids, torch.zeros_like(obj_ids), time, sequence
            )
            img = draw_boxes_in_image(
                img, to4x4(camera.camera_to_worlds), camera.get_intrinsics_matrices(), poses, dims, ids
            )
            return (img * 255).int().numpy()

        for frame in frames:
            anns = [ann for ann in annotations if ann.frame_id == frame and ann.sequence_id == self.sequence_id][0]
            if self.cameras is None:
                cam_images = {
                    im.cam_id: load_image(im, cameras[i], anns)
                    for i, im in enumerate(images)
                    if im.frame_id == frame and im.sequence_id == self.sequence_id and im.cam_id in cam_ids
                }
                output_frames.append(grid_cameras(cam_images, self.cameras_per_row))
            else:
                cam_images = {
                    self.cameras[int(im.cam_id)]: load_image(im, cameras[i], anns)
                    for i, im in enumerate(images)
                    if im.frame_id == frame and im.sequence_id == self.sequence_id and im.cam_id in cam_ids
                }
                if len(cam_images) > 1:
                    output_frames.append(tile_cameras(cam_images))
                else:
                    output_frames.append(list(cam_images.values())[0].astype(np.uint8))

        media.write_video(self.output_path / f"sequence_{self.sequence_id}.mp4", output_frames, fps=self.fps)


Commands = Union[
    Annotated[VisualizeScene, tyro.conf.subcommand(name="scene")],
    Annotated[VisualizePointcloud, tyro.conf.subcommand(name="pointcloud")],
    Annotated[VisualizeView, tyro.conf.subcommand(name="view")],
    Annotated[VisualizeVideo, tyro.conf.subcommand(name="video")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
