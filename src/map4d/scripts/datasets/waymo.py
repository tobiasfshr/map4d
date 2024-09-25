import io
import os
import pickle
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from tqdm import tqdm

from map4d.common.io import to_ply
from map4d.common.parallel import pmap
from map4d.data.parser.typing import AnnotationInfo, ImageInfo, PointCloudInfo

try:
    from waymo_open_dataset import dataset_pb2, label_pb2
    from waymo_open_dataset.utils import box_utils, range_image_utils, transform_utils
    from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
    from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
except ImportError:
    dataset_pb2 = None
    label_pb2 = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4])
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


@dataclass
class ProcessWaymo:
    """Process Waymo dataset. Uses utilities from https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md

    EmerNeRF uses the same split method, so the default train_split_fraction = 0.9 equals their NVS setting.
    See https://github.com/NVlabs/EmerNeRF/blob/8c051d7cccbad3b52c7b11a519c971b8ead97e1a/datasets/waymo.py#L511
    """

    data: Path = Path("data/waymo")
    """path to the data"""
    split_url: str = "assets/dynamic32.txt"
    """URL to the split file. Can also be local path."""
    scene_list_url: str = "assets/waymo_train_list.txt"
    """URL to the scene list file. Can also be local path."""
    num_workers: int = 16
    """The number of workers to use."""
    source: str = "gs://waymo_open_dataset_scene_flow/train"
    """The source directory to download the files from."""
    cameras: Tuple[str, ...] = ("FRONT_RIGHT", "FRONT_LEFT", "FRONT")
    """The cameras to use for the dataset."""
    image_hw: tuple[int, int] | None = (640, 960)
    """The image resolution to use for the dataset. If None, use original resolution."""
    max_distance: float = 0.0
    """The maximum distance to use for filtering the pointclouds and 3D boxes. If 0, no max distance."""
    use_second_return: bool = False
    """Whether to use the second return for the pointclouds."""
    rigid_classes: Tuple[str, ...] = "TYPE_VEHICLE"
    """The rigid classes to use for the dataset."""
    non_rigid_classes: Tuple[str, ...] = ("TYPE_PEDESTRIAN", "TYPE_CYCLIST")
    """The non-rigid classes to use for the dataset."""
    filter_no_label_zone_points: bool = False
    """Whether to filter lidar points within no label zones."""
    filter_empty_3dboxes: bool = True
    """Whether to filter out 3D boxes without any points."""

    def download_list(self, url: str) -> list[str]:
        """Load txt content from the url with requests"""
        if os.path.exists(url):
            with open(url, "r") as f:
                lines = f.readlines()
            content = [line.strip() for line in lines if not line.startswith("#")]
            return content

        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.split("\n")
            content = [line.strip() for line in lines if not line.startswith("#")]
            return content
        else:
            raise RuntimeError(f"Failed to download file from {url}. Status code: {response.status_code}")

    def _check_gcloud_status_ok(self):
        # check if gcloud sdk installed, download
        if not self.gcloud_status_ok:
            if not shutil.which("gcloud"):
                raise RuntimeError(
                    "gcloud not found. Please install the Google Cloud SDK, e.g. via `sudo snap install google-cloud-sdk`."
                )
            # check output of gcloud info
            result = subprocess.run(["gcloud", "info"], capture_output=True, text=True, check=True)
            result = result.stdout.split("Account: ")[1].split("\n")[0]
            if result == "[None]":
                raise RuntimeError("gcloud not authenticated. Please authenticate via `gcloud auth login`.")
            self.gcloud_status_ok = True

    def _download_file(self, scene):
        if not (self.data / "raw" / f"{scene}.tfrecord").exists():
            self._check_gcloud_status_ok()
            result = subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    "-n",
                    f"{self.source}/{scene}.tfrecord",
                    self.data / "raw",
                ],
                capture_output=True,  # To capture stderr and stdout for detailed error information
                text=True,
                check=True,  # To raise an exception if the command fails
            )

            # Check the return code of the gsutil command
            if result.returncode != 0:
                raise Exception(result.stderr)
        return self.data / "raw" / f"{scene}.tfrecord"

    def download_files(
        self,
        scene_names: List[str],
    ) -> list[Path]:
        """Downloads a list of files from a given source to a target directory using multiple threads."""
        # Get the total number of file_names
        total_files = len(scene_names)
        CONSOLE.log(f"Downloading {total_files} scenes from {self.source} to {self.data / 'raw'}")

        # Use ThreadPoolExecutor to manage concurrent downloads
        scene_files = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._download_file, scene) for scene in scene_names]
            for counter, future in enumerate(futures, start=1):
                # Wait for the download to complete and handle any exceptions
                try:
                    # inspects the result of the future and raises an exception if one occurred during execution
                    scene_path = future.result()
                    CONSOLE.log(f"[{counter}/{total_files}] Downloaded successfully!")
                    scene_files.append(scene_path)
                except Exception as e:
                    CONSOLE.log(f"[{counter}/{total_files}] Failed to download. Error: {e}")

        assert len(scene_files) == len(scene_names), "Not all files were downloaded."
        return scene_files

    def main(self):
        if dataset_pb2 is None:
            CONSOLE.log(
                "Waymo API is not installed. Please install it with `pip install waymo-open-dataset-tf-2-11-0==1.6.1 --no-deps`."
                + "Note that due to a dependency conflict with numpy, you need to install e.g. tensorflow manually after.",
                style="bold red",
            )
            return

        # download the data if not yet there
        os.makedirs(self.data / "raw", exist_ok=True)

        # scene_id, seg_name, start_timestep, end_timestep, scene_type
        split_info = self.download_list(self.split_url)
        scene_ids = [int(line.strip().split(",")[0]) for line in split_info]
        total_scene_list = self.download_list(self.scene_list_url)
        scene_names = [total_scene_list[i].strip() for i in scene_ids]

        self.gcloud_status_ok = False
        scene_files = self.download_files(scene_names)

        # process the data
        if tf is None:
            CONSOLE.log(
                "Tensorflow is not installed. Please install it with `pip install tensorflow`.", style="bold red"
            )
            return
        os.makedirs(self.data / "processed", exist_ok=True)
        pmap(self.process_sequence, zip(scene_names, scene_files), nprocs=self.num_workers)

    def process_sequence(self, sequence: str, file: Path):
        # get images / pcs / annotations / bounds
        images, pointclouds, annotations = [], [], []
        bounds = np.array([np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf])
        dataset = tf.data.TFRecordDataset(file, compression_type="")
        num_frames = sum(1 for _ in dataset)
        object_ids = []
        inv_first_ego_pose = None
        first_timestamp = None
        for frame_idx, data in enumerate(
            tqdm(dataset, desc=f"Sequence: {sequence}", total=num_frames, dynamic_ncols=True)
        ):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if inv_first_ego_pose is None:
                inv_first_ego_pose = np.linalg.inv(np.array(frame.pose.transform).reshape(4, 4))
            # relative pose to first frame
            ego_pose = inv_first_ego_pose @ np.array(frame.pose.transform).reshape(4, 4)
            if first_timestamp is None:
                first_timestamp = frame.timestamp_micros
            ims = self.get_images(frame, inv_first_ego_pose, first_timestamp, sequence, frame_idx)
            pointcloud = self.get_pointcloud(frame, ego_pose, first_timestamp, bounds, sequence, frame_idx)
            annotation = self.get_annotations(frame, ego_pose, first_timestamp, object_ids, frame_idx)
            self.get_dynamic_masks(frame, ims, sequence, frame_idx)
            images.extend(ims)
            pointclouds.append(pointcloud)
            annotations.append(annotation)

        assert set(im.frame_id for im in images) == set(
            ann.frame_id for ann in annotations
        ), "Missing annotations for a frame"

        # add some vertical space to bounds bc LiDAR in Waymo is quite narrow
        bounds[2] -= 1.0
        bounds[5] += 10.0
        CONSOLE.log("Final scene bounds:", bounds)
        depth_unit_scale_factor = 1.0  # TODO change when depth image is added
        with open(self.data / f"metadata_{sequence}.pkl", "wb") as f:
            pickle.dump((images, annotations, bounds, pointclouds, depth_unit_scale_factor), f)
        CONSOLE.log(f"Saved metadata to {self.data / f'metadata_{sequence}.pkl'}", style="bold green")

    def get_images(
        self, frame, inv_first_ego_pose: np.ndarray, first_timestamp, sequence: str, frame_idx: int
    ) -> list[ImageInfo]:
        images = []
        # opengl: x right, y up, z backward
        # waymo: x forward, y left, z up
        opengl_to_waymo_cam = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        for camera in frame.context.camera_calibrations:
            camera_name = dataset_pb2.CameraName.Name.Name(camera.name)
            if camera_name not in self.cameras:
                continue
            # NOTE: images and cameras are not aligned, need to find the correct image
            image = [image for image in frame.images if image.name == camera.name][0]
            if camera_name == "UNKNOWN":
                raise ValueError("Unknown camera name.")
            # camera parameters, note that waymo has a different cam coord convention than opencv
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            intrinsic_matrix = np.eye(3)
            intrinsic_matrix[0, 0] = list(camera.intrinsic)[0]
            intrinsic_matrix[1, 1] = list(camera.intrinsic)[1]
            intrinsic_matrix[0, 2] = list(camera.intrinsic)[2]
            intrinsic_matrix[1, 2] = list(camera.intrinsic)[3]

            # distortion parameters
            # NOTE: Actually it would be correct to undistort theimages but this is not done in EmerNeRF
            # k1 = camera.intrinsic[4]
            # k2 = camera.intrinsic[5]
            # k3 = camera.intrinsic[6]
            # k4 = camera.intrinsic[7]
            # k5 = camera.intrinsic[8]
            # save out image
            img_fpath = self.data / "processed" / sequence / "images" / camera_name / f"{frame_idx:06d}.jpg"
            os.makedirs(img_fpath.parent, exist_ok=True)

            # resize, save image
            img = Image.open(io.BytesIO(image.image))
            width, height = img.size
            if self.image_hw is not None:
                img = img.resize(self.image_hw[::-1], Image.BILINEAR)
                # scale in intrinsic matrix
                intrinsic_matrix[0] *= self.image_hw[1] / camera.width
                intrinsic_matrix[1] *= self.image_hw[0] / camera.height
                height, width = self.image_hw
            img.save(img_fpath)

            ego_cam_pose = inv_first_ego_pose @ np.array(image.pose.transform).reshape(4, 4)
            # mid exposure time, to microseconds, then to milliseconds
            timestamp = (image.pose_timestamp * 1e6 - first_timestamp) * 1e-3
            im = ImageInfo(
                image_path=img_fpath,
                width=width,
                height=height,
                pose=ego_cam_pose @ extrinsic @ opengl_to_waymo_cam,
                intrinsics=intrinsic_matrix,
                cam_id=self.cameras.index(camera_name),
                sequence_id=0,
                frame_id=frame_idx,
                timestamp=timestamp,
                depth_path=None,  # TODO add if you want to train with depth supervision
                flow_neighbors=None,
                mask_path=None,
            )
            images.append(im)
        return images

    def get_annotations(
        self, frame, ego_pose, first_timestamp, object_ids: list[str], frame_idx: int
    ) -> AnnotationInfo:
        """Extract annotation infos."""
        boxes, box_ids, class_ids = [], [], []
        for obj in frame.laser_labels:
            obj_type = label_pb2.Label.Type.Name(obj.type)
            if obj_type not in self.rigid_classes and obj_type not in self.non_rigid_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            # filter objects behind the ego vehicle
            if obj.box.center_x < 0.0:
                continue

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z
            width = obj.box.width
            length = obj.box.length
            height = obj.box.height
            # subtract 90 degrees since neutral position in waymo is along horizontal axis, not lateral axis
            rotation_z = obj.box.heading - np.pi / 2
            track_id = obj.id
            obj_class = 0 if obj_type in self.rigid_classes else 1
            box_params = np.array([x, y, z, width, length, height, rotation_z])

            # transform to global
            box_params[:3] = ego_pose[:3, :3] @ box_params[:3] + ego_pose[:3, 3]
            # add ego pose rotation z
            box_params[6] += np.arctan2(ego_pose[1, 0], ego_pose[0, 0])

            if track_id not in object_ids:
                object_ids.append(track_id)

            boxes.append(box_params)
            box_ids.append(object_ids.index(track_id))
            class_ids.append(obj_class)

        return AnnotationInfo(
            sequence_id=0,
            frame_id=frame_idx,
            # micro to milliseconds
            timestamp=(frame.timestamp_micros - first_timestamp) * 1e-3,
            boxes=np.array(boxes),
            box_ids=np.array(box_ids),
            class_ids=np.array(class_ids),
        )

    def get_dynamic_masks(self, frame, images: list[ImageInfo], sequence: str, frame_idx: int):
        """Save out dynamic masks for the frame, add to image info."""
        for img_info in images:
            filter_available = any([label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels])
            calibration = next(
                cc
                for cc in frame.context.camera_calibrations
                if dataset_pb2.CameraName.Name.Name(cc.name) == self.cameras[img_info.cam_id]
            )
            dynamic_mask = np.zeros((calibration.height, calibration.width), dtype=np.float32)
            for label in frame.laser_labels:
                # camera_synced_box is not available for the data with flow.
                # box = label.camera_synced_box
                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                if not box.ByteSize():
                    continue  # Filter out labels that do not have a camera_synced_box.
                if (filter_available and not label.num_top_lidar_points_in_box) or (
                    not filter_available and not label.num_lidar_points_in_box
                ):
                    continue  # Filter out likely occluded objects.

                # Retrieve upright 3D box corners.
                box_coords = np.array(
                    [
                        [
                            box.center_x,
                            box.center_y,
                            box.center_z,
                            box.length,
                            box.width,
                            box.height,
                            box.heading,
                        ]
                    ]
                )
                corners = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()  # [8, 3]

                # Project box corners from vehicle coordinates onto the image.
                projected_corners = project_vehicle_to_image(frame.pose, calibration, corners)
                u, v, ok = projected_corners.transpose()
                ok = ok.astype(bool)

                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u = u[ok]
                v = v[ok]

                # Clip box to image bounds.
                u = np.clip(u, 0, calibration.width)
                v = np.clip(v, 0, calibration.height)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    speed,
                )
            # thresholding, use 1.0 m/s to determine whether the pixel is moving
            dynamic_mask_path = (
                self.data
                / "processed"
                / sequence
                / "dynamic_masks"
                / self.cameras[img_info.cam_id]
                / f"{frame_idx:06d}.png"
            )
            os.makedirs(dynamic_mask_path.parent, exist_ok=True)
            dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
            dynamic_mask = Image.fromarray(dynamic_mask, "L")
            if self.image_hw is not None:
                dynamic_mask = dynamic_mask.resize(self.image_hw[::-1], Image.BILINEAR)
                dynamic_mask = np.clip((np.array(dynamic_mask) > 0) * 255, 0, 255).astype(np.uint8)
                dynamic_mask = Image.fromarray(dynamic_mask, "L")
            dynamic_mask.save(dynamic_mask_path)
            img_info.mask_path = dynamic_mask_path

    def get_pointcloud(
        self, frame, ego_pose: np.ndarray, first_timestamp, bounds: np.ndarray, sequence: str, frame_idx: int
    ) -> PointCloudInfo:
        """Extract pointcloud, save it to ply and return info."""
        range_images, _, _, range_image_top_pose = parse_range_image_and_camera_projection(frame)
        assert range_image_top_pose is not None, "No LiDAR pointcloud"
        # First return
        points = self.range_images_to_pointcloud(frame, range_images, range_image_top_pose, ri_index=0)

        if self.use_second_return:
            points_1 = self.range_images_to_pointcloud(frame, range_images, range_image_top_pose, ri_index=1)
            points = np.concatenate([points, points_1], axis=0)

        # delete points behind the ego-vehicle
        points = points[points[:, 0] > 0]

        # transform points to world
        points = points @ ego_pose[:3, :3].T + ego_pose[:3, 3]

        # update bounds
        pc_min, pc_max = np.min(points, axis=0), np.max(points, axis=0)
        bounds[:3] = np.minimum(bounds[:3], pc_min)
        bounds[3:] = np.maximum(bounds[3:], pc_max)

        # Save out pointcloud
        pc_fpath = self.data / "processed" / sequence / "pointcloud" / f"{frame_idx:06d}.ply"
        os.makedirs(pc_fpath.parent, exist_ok=True)
        to_ply(pc_fpath, points)
        # TODO not saving colors (not needed when training with neural fields).
        return PointCloudInfo(
            points_path=pc_fpath,
            sequence_id=0,
            frame_id=frame_idx,
            timestamp=(frame.timestamp_micros - first_timestamp) * 1e-3,
        )

    def range_images_to_pointcloud(self, frame, range_images, range_image_top_pose, ri_index):
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = []

        frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data), range_image_top_pose.shape.dims
        )
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2],
        )
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
        )
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]), height=range_image.shape.dims[0]
                )
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask
            mask_index = tf.where(range_image_mask)

            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local,
            )
            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)
            points.append(points_tensor.numpy())

        return np.concatenate(points, axis=0)
