"""Render out a trajectory from a camera path."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.camera_utils import get_interpolated_poses
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.scripts.render import BaseRender, _render_trajectory_video, get_crop_from_json
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from typing_extensions import Annotated

from map4d.common.visualize import apply_depth_colormap
from map4d.scripts.util import grid_cameras, tile_cameras


def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


def generate_trajectory(
    train_dataparser_outputs: DataparserOutputs,
    test_dataparser_outputs: DataparserOutputs,
    camera_id: int = 0,
    sequence_id: int = 0,
    smooth: bool = False,
    rotate: bool = False,
    mirror: bool = False,
    image_height: int = 1080,
    image_width: int = 1920,
    fov: int = 50,
) -> Cameras:
    """Generate a (smoothed) trajectory from the camera path.

    If smoothed, we interpolate between keyframes and add rotation
    and translation, and mirror the trajectory at the middle.

    Args:
        train_dataparser_outputs: train dataparser outputs
        test_dataparser_outputs: test dataparser outputs
        camera_id: the camera id
        sequence_id: the sequence id
        smooth: whether to smooth the trajectory
        rotate: whether to rotate the camera
        image_height: the image height
        image_width: the image width
        fov: the field of view

    Returns:
        Cameras: the cameras to render.
    """
    cams = []
    for i, cam in enumerate(train_dataparser_outputs.cameras):
        if cam.metadata["camera_ids"] == camera_id and cam.metadata["sequence_ids"] == sequence_id:
            cams.append(cam)

    for i, cam in enumerate(test_dataparser_outputs.cameras):
        if cam.metadata["camera_ids"] == camera_id and cam.metadata["sequence_ids"] == sequence_id:
            cams.append(cam)

    cams = sorted(cams, key=lambda x: x.times[0])

    if not (smooth or rotate):
        cam2worlds = torch.stack([cam.camera_to_worlds for cam in cams])
        fxs = torch.stack([cam.fx for cam in cams])
        fys = torch.stack([cam.fy for cam in cams])
        cxs = torch.stack([cam.cx for cam in cams])
        cys = torch.stack([cam.cy for cam in cams])
        times = torch.stack([cam.times for cam in cams])
        cam_idx = torch.stack([cam.metadata["cam_idx"] for cam in cams])
        camera_ids = torch.stack([cam.metadata["camera_ids"] for cam in cams])
        sequence_ids = torch.stack([cam.metadata["sequence_ids"] for cam in cams])
        image_height = cams[0].height
        image_width = cams[0].width
        cams = Cameras(
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            height=image_height,
            width=image_width,
            camera_to_worlds=cam2worlds,
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"camera_ids": camera_ids, "sequence_ids": sequence_ids, "cam_idx": cam_idx},
        )
        return cams

    cam2worlds = torch.stack([cam.camera_to_worlds for cam in cams])
    fxs, fys = [], []
    if smooth:
        dataparser_scale = train_dataparser_outputs.dataparser_scale
        # keyframe selection
        keyframes = [cam2worlds[0]]
        for c2w in cam2worlds[1:]:
            # if translation is larger than 1m, add as keyframe
            if torch.linalg.norm(c2w[:3, 3] - keyframes[-1][:3, 3]) > 1.0 * dataparser_scale:
                keyframes.append(c2w)

        keyframes = torch.stack(keyframes)

        # interpolate trajectory from keyframes
        steps_per_transition = 10
        traj = []
        for idx in range(keyframes.shape[0] - 1):
            pose_a = keyframes[idx]
            pose_b = keyframes[idx + 1]
            poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)[:-1]
            traj += poses_ab
        cam2worlds = np.stack(traj, axis=0)
        cam2worlds = torch.from_numpy(cam2worlds).float()[:, :3]
        focal_length = three_js_perspective_camera_focal_length(fov, image_height)
        fxs.append(focal_length)
        fys.append(focal_length)

        # mirror trajectory at middle
        if mirror:
            cam2worlds[len(cam2worlds) // 2 :] = cam2worlds[: (len(cam2worlds) // 2 + len(cam2worlds) % 2)].flip(0)

        # rotate camera
        if rotate:
            rot_angles_z = np.linspace(0, 1.0, len(cam2worlds)) * -np.pi * 2  # turn 360 clockwise
            rot_angles_x = np.sin(np.linspace(0, 2 * np.pi, len(cam2worlds))) * np.pi / 8  # 22.5 forth and back
            rot_angles_y = np.zeros_like(rot_angles_z)
            for i, (rot_x, rot_y, rot_z) in enumerate(zip(rot_angles_x, rot_angles_y, rot_angles_z)):
                rot_mat_x = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, np.cos(rot_x), -np.sin(rot_x)], [0.0, np.sin(rot_x), np.cos(rot_x)]]
                ).float()

                rot_mat_y = torch.tensor(
                    [[np.cos(rot_y), 0.0, np.sin(rot_y)], [0.0, 1.0, 0.0], [-np.sin(rot_y), 0.0, np.cos(rot_y)]]
                ).float()

                rot_mat_z = torch.tensor(
                    [[np.cos(rot_z), -np.sin(rot_z), 0.0], [np.sin(rot_z), np.cos(rot_z), 0.0], [0.0, 0.0, 1.0]]
                ).float()
                cam2worlds[i, :3, :3] = rot_mat_z @ rot_mat_y @ rot_mat_x @ cam2worlds[i, :3, :3]

        # alter the to the z coordinate 50m up / down
        cam2worlds[:, 2, 3] += np.sin(np.linspace(0, np.pi * 2, len(cam2worlds))) * 0.5 * dataparser_scale

    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    times = torch.pow(torch.linspace(-1.0, 1.0, len(cam2worlds)), 3).unsqueeze(-1).float()
    sequence_ids = torch.ones_like(times) * sequence_id

    cams = Cameras(
        fx=fx,
        fy=fy,
        cx=image_width / 2,
        cy=image_height / 2,
        height=image_height,
        width=image_width,
        camera_to_worlds=cam2worlds,
        camera_type=CameraType.PERSPECTIVE,
        times=times,
        metadata={"sequence_ids": sequence_ids},
    )

    return cams


@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    camera_path_filename: Optional[Path] = None
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    camera_id: int | tuple[int, ...] = 0
    """The camera id to render. To render multiple cameras next to each other, pass a tuple."""
    fps: int = 10
    """Frames per second for the output video."""
    cameras: tuple[str, ...] | None = None
    """The camera names for tile_cameras. Use None to align cameras in a grid."""
    cameras_per_row: int = 4
    """Number of cameras per row for the grid layout."""
    smooth: bool = False
    """Whether to smooth the camera path."""
    rotate: bool = False
    """Whether to rotate the camera."""
    mirror: bool = False
    """Whether to mirror the camera path in the middle."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions("inferno_r")
    """Options for the colormap."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        install_checks.check_ffmpeg_installed()

        if self.camera_path_filename is None:
            parser = pipeline.datamanager.dataparser
            dpo = parser.get_dataparser_outputs(split="train")
            test_dpo = parser.get_dataparser_outputs(split="test")
            if isinstance(self.camera_id, int):
                self.camera_id = (self.camera_id,)
            camera_paths = []
            for cam_id in self.camera_id:
                camera_path = generate_trajectory(
                    dpo, test_dpo, camera_id=cam_id, smooth=self.smooth, rotate=self.rotate, mirror=self.mirror
                )
                camera_paths.append(camera_path)

            assert all([len(camera_paths[0]) == len(camera_path) for camera_path in camera_paths])
            crop_data = None
            seconds = len(camera_paths[0]) / self.fps

            if len(camera_paths) == 1:
                camera_path = camera_paths[0]
            else:
                camera_path = None
        else:
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            crop_data = get_crop_from_json(camera_path)
            camera_path = get_path_from_json(camera_path)
            train_times = pipeline.datamanager.train_dataset.cameras.times
            camera_path.times = (
                torch.linspace(0.0, 1.0, len(camera_path)) * (train_times.max() - train_times.min()) + train_times.min()
            )

        min_d, max_d = pipeline.model.config.min_depth, pipeline.model.config.max_depth
        min_d, max_d = min_d * pipeline.model.dataparser_scale, max_d * pipeline.model.dataparser_scale

        if camera_path is not None:
            camera_path.metadata["render_images"] = False
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                colormap_options=self.colormap_options,
                depth_near_plane=min_d,
                depth_far_plane=max_d,
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                for cam_id, camera_path in zip(self.camera_id, camera_paths):
                    camera_path.metadata["render_images"] = False
                    _render_trajectory_video(
                        pipeline,
                        camera_path,
                        output_filename=Path(temp_dir + f"/{cam_id}"),
                        rendered_output_names=self.rendered_output_names,
                        rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                        crop_data=crop_data,
                        seconds=seconds,
                        output_format="images",
                        colormap_options=self.colormap_options,
                        depth_near_plane=min_d,
                        depth_far_plane=max_d,
                    )
                frames = os.listdir(temp_dir + f"/{self.camera_id[0]}")

                if self.output_format == "images":
                    output_path = self.output_path.parent / self.output_path.stem
                    output_path.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = self.output_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                video_frames = []
                for frame in sorted(frames):
                    images = {name: {} for name in self.rendered_output_names}
                    if self.cameras is None:
                        for cam_id in self.camera_id:
                            image = media.read_image(temp_dir + f"/{cam_id}/{frame}")
                            im_w, offset = image.shape[1] // len(self.rendered_output_names), 0
                            for output in self.rendered_output_names:
                                images[output][f"{cam_id}"] = image[:, offset : offset + im_w]
                                offset += im_w
                        grid_ims = {}
                        for output in self.rendered_output_names:
                            grid_ims[output] = grid_cameras(images[output], self.cameras_per_row)
                    else:
                        for cam_name, cam_id in zip(self.camera_id, self.cameras):
                            image = media.read_image(temp_dir + f"/{cam_id}/{frame}")
                            im_w, offset = image.shape[1] // len(self.rendered_output_names), 0
                            for output in self.rendered_output_names:
                                images[output][f"{cam_name}"] = image[:, offset : offset + im_w]
                                offset += im_w
                        grid_ims = {}
                        for output in self.rendered_output_names:
                            grid_ims[output] = tile_cameras(images[output])

                    if self.output_format != "video":
                        for name in self.rendered_output_names:
                            media.write_image(output_path / frame.replace(".", f"_{name}."), grid_ims[name])
                    else:
                        video_frames.append(grid_ims)
                if self.output_format == "video":
                    for name in self.rendered_output_names:
                        vid_frames = [frame[name] for frame in video_frames]
                        media.write_video(str(output_path).replace(".mp4", f"_{name}.mp4"), vid_frames, fps=self.fps)
            CONSOLE.print(f"Output written to {output_path}.", justify="center")


@dataclass
class ViewRender(BaseRender):
    """Render a certain evaluation image given its index."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc."""
    image_idx: Union[int, List[int]] = -1
    """Index of the image to render."""
    output_path: Path = Path("renders")
    """Path to output directory."""
    render_debugging_images: bool = False
    """Whether to render debugging images."""

    split: Literal["train", "test"] = "test"

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        if self.split == "train":
            cameras = pipeline.datamanager.train_dataset.cameras
        else:
            cameras = pipeline.datamanager.eval_dataset.cameras
        cameras.rescale_output_resolution(1.0 / self.downscale_factor)
        cameras = cameras.to(pipeline.device)
        pipeline.train(self.split == "train")

        min_d, max_d = pipeline.model.config.min_depth, pipeline.model.config.max_depth
        min_d, max_d = min_d * pipeline.model.dataparser_scale, max_d * pipeline.model.dataparser_scale

        if self.image_idx == -1:
            self.image_idx = list(range(len(cameras)))
        elif isinstance(self.image_idx, int):
            self.image_idx = [self.image_idx]

        os.makedirs(self.output_path, exist_ok=True)
        for im_idx in self.image_idx:
            for rendered_output_name in self.rendered_output_names:
                with torch.no_grad():
                    cam = cameras[im_idx : im_idx + 1]
                    cam.metadata["cam_idx"] = torch.tensor([im_idx])
                    cam.metadata["render_images"] = self.render_debugging_images
                    outputs = pipeline.model.get_outputs_for_camera(cam)

                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name]
                if rendered_output_name == "depth":
                    output_image = apply_depth_colormap(output_image, min_d, max_d)
                    output_image = output_image.cpu().numpy()
                else:
                    output_image = colormaps.apply_colormap(image=output_image).cpu().numpy()

                outpath = (
                    self.output_path
                    / f"{im_idx:05d}_{rendered_output_name}.{'png' if self.image_format == 'png' else 'jpg'}"
                )
                media.write_image(outpath, output_image)
                CONSOLE.print(f"Output written to {outpath}.", justify="center")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[ViewRender, tyro.conf.subcommand(name="view")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
