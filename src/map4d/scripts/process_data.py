import io
import os
import pickle
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
import tyro
import utm
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from typing_extensions import Annotated

from map4d.common.geometry import depth_to_points, opencv_to_opengl
from map4d.common.io import get_depth_image_from_path, to_ply
from map4d.common.parallel import pmap
from map4d.common.pointcloud import transform_points
from map4d.data.parser.typing import AnnotationInfo, ImageInfo, PointCloudInfo

try:
    import av2
    import av2.utils.io as io_utils
    from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader, convert_pose_dataframe_to_SE3
    from av2.geometry.se3 import SE3
    from av2.structures.cuboid import CuboidList
except ImportError:
    av2 = None


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


def frame_in_range(frame, frame_ranges):
    if isinstance(frame_ranges[0], list):
        for fr_ra in frame_ranges:
            if fr_ra[0] <= frame <= fr_ra[1]:
                return True
    else:
        return frame_ranges[0] <= frame <= frame_ranges[1]
    return False


@dataclass
class ProcessArgoverse2:
    cameras: Tuple[str, ...] = (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    )
    """The cameras to use for the dataset."""
    data: Path = Path("data/Argoverse2/")
    """Directory specifying location of data."""
    city: str = "PIT"
    """one of WDC, PIT, PAO, ATX, MIA, DTW"""
    location_aabb: tuple[int, int, int, int] | None = None
    """The location AABB to use for filtering the data. (xmin, ymin, xmax, ymax)"""
    log_id: str | None = "0c61aea3-3cba-35f3-8971-df42cd5b9b1a"
    """The log to use for the dataset."""
    max_distance: float = 80.0
    """The maximum distance to use for filtering the pointclouds and 3D boxes. If 0, no max distance."""
    rigid_classes: Tuple[str, ...] = (
        "REGULAR_VEHICLE",
        "LARGE_VEHICLE",
        "BUS",
        "ARTICULATED_BUS",
        "BOX_TRUCK",
        "TRUCK",
        "TRUCK_CAB",
        "VEHICULAR_TRAILER",
        "SCHOOL_BUS",
        "MESSAGE_BOARD_TRAILER",
        "STROLLER",
        "VEHICULAR_TRAILER",
        "WHEELED_DEVICE",
    )
    """The rigid classes to use for the dataset (excl. motorcycle, bicycle, wheelchair)."""
    nonrigid_classes: Tuple[str, ...] = ("PEDESTRIAN", "BICYCLIST", "MOTORCYCLIST", "DOG", "WHEELED_RIDER")
    """The non-rigid classes to use for the dataset."""
    align_pointclouds: bool = True
    """Whether to align the pointclouds of multiple sequences with alignment file."""
    alignment_path: Path = Path("assets/")
    """The alignment path to use for aligning the pointclouds."""
    box_preds_path: Path | None = None
    """The path to the box predictions. If None, use the ground truth boxes."""
    split: Literal["train", "val", "test"] = "train"
    """The split to use for the dataset."""
    masks_path: Path = Path("assets/masks").absolute()
    """Path to ego-vehicle masks."""

    def main(self):
        if av2 is None:
            CONSOLE.log(
                "AV2 API is not installed. Please install it with `pip install git+https://github.com/tobiasfshr/av2-api.git`.",
                style="bold red",
            )
            return
        self.prepare_seq()

    def prepare_seq(self, seq_name: str | None = None):
        if self.location_aabb is not None:
            assert self.location_aabb is not None and self.city is not None
            seq_name = self.city + "_" + "_".join(map(str, self.location_aabb))
            log_id = None
        elif seq_name is None:
            log_id = self.log_id
            seq_name = log_id
        else:
            log_id = seq_name

        # use always train since test are different sequences
        self.loader = AV2SensorDataLoader(data_dir=self.data / self.split, labels_dir=self.data)

        # get relevant log ids, timestamps
        CONSOLE.log("getting log ids and timestamps")
        split_log_ids, timestamps_per_log = self.get_logs_timestamps(log_id)

        # compute pointcloud, alignment (possibly), depth bounds
        global_pointclouds, transforms_per_log, bounds = self.compute_pointclouds_alignment_bounds(
            seq_name, split_log_ids, timestamps_per_log
        )

        # generate image, annotation info
        CONSOLE.log("generating image, annotation info")
        images: list[ImageInfo] = []
        annotations: list[AnnotationInfo] = []
        ego_masks = {}
        for camera in self.cameras:
            ego_masks[camera] = self.masks_path / (camera + ".png")
            assert ego_masks[camera].exists(), f"Ego vehicle mask not found: {ego_masks[camera]}"
        all_object_ids = []
        for i, (log, timestamps, align_transform) in enumerate(
            zip(split_log_ids, timestamps_per_log, transforms_per_log)
        ):
            log_poses_df = io_utils.read_feather(self.data / self.split / log / "city_SE3_egovehicle.feather")
            cam_models = {cam: self.loader.get_log_pinhole_camera(log, cam) for cam in self.cameras}

            preds_log = None
            if self.box_preds_path is not None:
                preds_log = pickle.load(open(self.box_preds_path / f"preds_{log}.pkl", "rb"))

            CONSOLE.log(f"Processing log {log}. {i+1}/{len(split_log_ids)}")
            for frame_id, timestamp in tqdm(enumerate(timestamps)):
                lidar_pose = self.get_city_SE3_ego(log_poses_df, timestamp)

                if preds_log is not None:
                    pred = preds_log[timestamp]
                    boxes, box_ids = pred["bboxes"], pred["object_ids"]
                    for i, id in enumerate(box_ids):
                        if f"{log}_{id}" not in all_object_ids:
                            all_object_ids.append(f"{log}_{id}")
                    # make prediction ids unique across sequences
                    box_ids = np.array([all_object_ids.index(f"{log}_{id}") for id in box_ids])
                    class_ids = np.zeros_like(box_ids)
                else:
                    frame_anns = self.loader.get_labels_at_lidar_timestamp(log, timestamp)
                    # NOTE: adding track id to cuboid has side effects on those methods, added fix for transform method only
                    frame_anns = frame_anns.transform(lidar_pose)
                    frame_anns = self.filter_boxes(frame_anns, bounds)

                    # transform with alignment transform
                    align_transform_se3 = SE3(align_transform[:3, :3], align_transform[:3, 3])
                    frame_anns = frame_anns.transform(align_transform_se3)

                    # extract boxes
                    boxes, box_ids, class_ids = self.get_boxes(frame_anns, all_object_ids)

                vehicle2world = align_transform @ lidar_pose.transform_matrix
                ann_info = AnnotationInfo(
                    sequence_id=split_log_ids.index(log),
                    frame_id=frame_id,
                    timestamp=timestamp - timestamps[0],
                    vehicle2world=vehicle2world,
                    boxes=boxes,
                    box_ids=box_ids,
                    class_ids=class_ids,
                )
                annotations.append(ann_info)

                img_fpaths = []
                for camera in self.cameras:
                    img_fpath = self.loader.get_closest_img_fpath(log, camera, timestamp)
                    assert img_fpath is not None
                    img_fpaths.append(img_fpath)

                for camera, img_fpath in zip(self.cameras, img_fpaths):
                    cam_model = cam_models[camera]
                    cam_intr = cam_model.intrinsics.K

                    depth_fpath = (
                        self.data
                        / "depth"
                        / self.split
                        / seq_name
                        / log
                        / camera
                        / (img_fpath.parts[-1].replace(".jpg", ".parquet"))
                    )
                    cam_timestamp = int(img_fpath.parts[-1].replace(".jpg", ""))
                    cam_pose = self.get_city_SE3_ego(log_poses_df, cam_timestamp)

                    # image to world
                    cam2vehicle = (
                        lidar_pose.inverse().transform_matrix
                        @ cam_pose.transform_matrix
                        @ cam_model.ego_SE3_cam.transform_matrix
                        @ opencv_to_opengl
                    )
                    pose = (
                        align_transform
                        @ cam_pose.transform_matrix
                        @ cam_model.ego_SE3_cam.transform_matrix
                        @ opencv_to_opengl
                    )

                    if not os.path.exists(depth_fpath):
                        self.save_depth_map(depth_fpath, log, timestamp, cam_timestamp, camera, cam_model)

                    img_info = ImageInfo(
                        image_path=img_fpath,
                        width=cam_model.width_px,
                        height=cam_model.height_px,
                        pose=pose,
                        cam2vehicle=cam2vehicle,
                        intrinsics=cam_intr,
                        mask_path=ego_masks[camera],
                        depth_path=depth_fpath,
                        cam_id=self.cameras.index(camera),
                        sequence_id=split_log_ids.index(log),
                        frame_id=frame_id,
                        timestamp=cam_timestamp - timestamps[0],
                        flow_neighbors=None,
                    )
                    images.append(img_info)

        assert set(im.frame_id for im in images) == set(
            ann.frame_id for ann in annotations
        ), "Missing annotations for a frame"

        depth_unit_scale_factor = 1.0
        save_path = (
            self.data / f"metadata_{seq_name}.pkl"
            if self.box_preds_path is None
            else self.data / f"metadata_{seq_name}_preds.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump((images, annotations, bounds, global_pointclouds, depth_unit_scale_factor), f)
        CONSOLE.log(f"Saved metadata to {save_path}", style="bold green")

    def compute_pointclouds_alignment_bounds(self, seq_name, log_ids, timestamps_per_log):
        if len(log_ids) > 1 and self.align_pointclouds:
            assert os.path.exists(
                self.alignment_path / f"{seq_name}_transforms.pkl"
            ), f"Alignment file {self.alignment_path / f'{seq_name}_transforms.pkl'} does not exist."
            transforms_per_seq = pickle.load(open(self.alignment_path / f"{seq_name}_transforms.pkl", "rb"))
        else:
            transforms_per_seq = [np.eye(4) for _ in range(len(log_ids))]

        CONSOLE.log("Computing global pointclouds...")
        global_pcs = []
        bounds = np.array([np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf])
        for log, timestamps, transform in tqdm(zip(log_ids, timestamps_per_log, transforms_per_seq)):
            log_poses_df = io_utils.read_feather(self.data / self.split / log / "city_SE3_egovehicle.feather")
            for timestamp in timestamps:
                # loader pointcloud at timestamp
                lidar_pose = self.get_city_SE3_ego(log_poses_df, timestamp)
                lidar_fpath = self.loader.get_closest_lidar_fpath(log, timestamp)
                lidar_pc = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")

                # get colors
                global_col = np.zeros_like(lidar_pc)
                hit_mask = np.zeros(lidar_pc.shape[0], dtype=bool)
                for camera in self.cameras:
                    img_fpath = self.loader.get_closest_img_fpath(log, camera, timestamp)
                    image = np.array(Image.open(img_fpath))
                    assert img_fpath is not None
                    cam_timestamp = int(img_fpath.parts[-1].replace(".jpg", ""))

                    # project lidar to image
                    uv, _, is_valid_points = self.loader.project_ego_to_img_motion_compensated(
                        points_lidar_time=lidar_pc,
                        cam_name=camera,
                        cam_timestamp_ns=cam_timestamp,
                        lidar_timestamp_ns=timestamp,
                        log_id=log,
                    )
                    is_valid_points = np.logical_and(is_valid_points, ~hit_mask)
                    uv = uv[is_valid_points]
                    global_col[is_valid_points] = image[
                        np.round(uv[:, 1]).astype(np.int32), np.round(uv[:, 0]).astype(np.int32)
                    ]
                    hit_mask[is_valid_points] = True

                # transform pointcloud to global
                global_pc = lidar_pose.transform_point_cloud(lidar_pc)

                # load annotations
                annotations = self.loader.get_labels_at_lidar_timestamp(log, timestamp)
                annotations = annotations.transform(lidar_pose)

                # apply sequence alignment
                global_pc = global_pc @ transform[:3, :3].T + transform[:3, 3]

                # update bounds (filtering for very far away points)
                point_mask = np.linalg.norm(lidar_pc, axis=1) < self.max_distance
                pc_min, pc_max = np.min(global_pc[point_mask], axis=0), np.max(global_pc[point_mask], axis=0)
                bounds[:3] = np.minimum(bounds[:3], pc_min)
                bounds[3:] = np.maximum(bounds[3:], pc_max)

                # save colored pointcloud to ply, create PointCloudInfo
                os.makedirs(self.data / "pointclouds" / log, exist_ok=True)
                to_ply(self.data / "pointclouds" / log / f"{timestamp}.ply", global_pc, global_col)

                pointcloud = PointCloudInfo(
                    points_path=self.data / "pointclouds" / log / f"{timestamp}.ply",
                    sequence_id=log_ids.index(log),
                    frame_id=timestamps.index(timestamp),
                    timestamp=timestamp - timestamps[0],
                )

                global_pcs.append(pointcloud)

        return global_pcs, transforms_per_seq, bounds

    def filter_boxes(self, annotations, bounds):
        anns_filtered = []
        for ann in annotations:
            # check if center is outside bounds
            if (ann.xyz_center_m < bounds[:3]).any() or (ann.xyz_center_m > bounds[3:]).any():
                CONSOLE.log(
                    f"Filtered out box {ann.track_id} at {ann.xyz_center_m}, bounds: {bounds}", style="bold yellow"
                )
                continue
            anns_filtered.append(ann)
        return CuboidList(cuboids=anns_filtered)

    def get_boxes(self, annotations, all_ids):
        """Get boxes from annotations in xyzwlhrz format."""
        boxes, box_ids, class_ids = [], [], []
        for ann in annotations:
            if (ann.category not in self.rigid_classes) and (ann.category not in self.nonrigid_classes):
                continue
            center = ann.xyz_center_m
            size = ann.dims_lwh_m[1], ann.dims_lwh_m[0], ann.dims_lwh_m[2]
            # subtract 90 degrees since neutral position in argoverse is along horizontal axis, not lateral axis
            yaw = R.from_matrix(ann.dst_SE3_object.rotation).as_euler("xyz")[2] - np.pi / 2
            box = [*center, *size, yaw]
            boxes.append(box)
            if ann.track_id not in all_ids:
                all_ids.append(ann.track_id)
            box_ids.append(all_ids.index(ann.track_id))
            class_ids.append(0 if ann.category in self.rigid_classes else 1)
        return np.array(boxes), np.array(box_ids), np.array(class_ids)

    def get_logs_timestamps(self, log_id: str | None = None):
        log_ids = self.loader.get_log_ids()
        timestamps_per_log = []
        split_log_ids = []
        for log in log_ids:
            log_city = self.loader.get_city_name(log)

            if self.city not in ("None", log_city):
                continue

            if log_id is not None and log != log_id:
                continue

            log_poses_df = io_utils.read_feather(self.data / self.split / log / "city_SE3_egovehicle.feather")
            timestamps = self.loader.get_ordered_log_lidar_timestamps(log)
            used_timestamps = []
            for timestamp in timestamps:
                # check if location AABB is set and if the current frame is in the AABB
                if self.location_aabb is not None:
                    lidar_pose = self.get_city_SE3_ego(log_poses_df, timestamp)
                    tx, ty, _ = lidar_pose.translation
                    if not (
                        self.location_aabb[0] < tx < self.location_aabb[2]
                        and self.location_aabb[1] < ty < self.location_aabb[3]
                    ):
                        continue

                # check if all cameras are available at timestamp
                img_fpaths = []
                for camera in self.cameras:
                    img_fpath = self.loader.get_closest_img_fpath(log, camera, timestamp)
                    if img_fpath is None:
                        CONSOLE.log(
                            f"Assumption that there is an image for all cameras is broken at: {timestamp}, {log}, {camera}. Skipping frame."
                        )
                        img_fpath = []
                        break
                    img_fpaths.append(img_fpath)
                if len(img_fpaths) < len(self.cameras):
                    continue

                # now we are sure the frame can be added
                used_timestamps.append(timestamp)

            # if a frame was added, add log to the list [contrary only happens if location_aabb is set]
            if len(used_timestamps) > 0:
                split_log_ids.append(log)
                timestamps_per_log.append(used_timestamps)

        return split_log_ids, timestamps_per_log

    def get_city_SE3_ego(self, log_poses_df, timestamp_ns):
        pose_df = log_poses_df.loc[log_poses_df["timestamp_ns"] == timestamp_ns]

        if len(pose_df) == 0:
            raise RuntimeError("Pose was not available for the requested timestamp.")

        city_SE3_ego = convert_pose_dataframe_to_SE3(pose_df)
        return city_SE3_ego

    def save_depth_map(self, depth_fpath: Path, log: str, timestamp: int, cam_timestamp: int, camera: str, cam_model):
        # load lidar
        lidar_fpath = self.loader.get_closest_lidar_fpath(log, timestamp)
        lidar_pc = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")

        # project lidar to image
        (
            uv,
            points_cam,
            is_valid_points,
        ) = self.loader.project_ego_to_img_motion_compensated(
            points_lidar_time=lidar_pc,
            cam_name=camera,
            cam_timestamp_ns=cam_timestamp,
            lidar_timestamp_ns=timestamp,
            log_id=log,
        )
        depth_map = np.zeros((cam_model.height_px, cam_model.width_px))
        uv, points_cam = uv[is_valid_points], points_cam[is_valid_points]
        depth_map[np.floor(uv[:, 1]).astype(np.int32), np.ceil(uv[:, 0]).astype(np.int32)] = points_cam[:, 2]
        depth_map[np.ceil(uv[:, 1]).astype(np.int32), np.ceil(uv[:, 0]).astype(np.int32)] = points_cam[:, 2]
        depth_map[np.floor(uv[:, 1]).astype(np.int32), np.floor(uv[:, 0]).astype(np.int32)] = points_cam[:, 2]
        depth_map[np.ceil(uv[:, 1]).astype(np.int32), np.floor(uv[:, 0]).astype(np.int32)] = points_cam[:, 2]

        # save depth map, use parquet for compression
        os.makedirs(depth_fpath.parent, exist_ok=True)
        pq.write_table(
            pa.table({"depth": depth_map.flatten()}, metadata={"shape": " ".join([str(x) for x in depth_map.shape])}),
            depth_fpath,
            filesystem=None,
            compression="BROTLI",
        )


KITTI_FRAME_RANGE = {"0001": [380, 431], "0002": [140, 224], "0006": [65, 120]}


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


@dataclass
class ProcessKITTI:
    data: Path = Path("data/KITTI/tracking/training")
    """path to the data"""
    sequence: Literal["0001", "0002", "0006"] = "0001"
    """sequence to use"""
    task: Literal["nvs", "imrec"] = "nvs"
    """task to evaluate. nvs: novel view synthesis, imrec: image reconstruction"""
    max_distance: float = 80.0
    """The maximum distance to use for filtering the pointclouds and 3D boxes. If 0, no max distance."""
    mars_split: bool = False
    """Whether to use the split from MARS for the dataset."""

    def main(self):
        if self.task == "imrec" and self.sequence == "0006":
            frame_range = [[0, 20], [65, 120]]
        else:
            frame_range = KITTI_FRAME_RANGE[self.sequence]

        if self.mars_split:
            assert self.sequence == "0006", "MARS evaluates only on sequence 0006."
            frame_range = [5, 260]

        calib: Dict[str, np.ndarray] = {}
        with open("{}/calib/{}.txt".format(self.data, self.sequence), "r") as f:
            for line in f:
                tokens = line.strip().split()
                calib[tokens[0]] = np.array([float(x) for x in tokens[1:]])

        imu2velo = np.eye(4)
        imu2velo[:3] = calib["Tr_imu_velo"].reshape(3, 4)

        velo2cam_base = np.eye(4)
        velo2cam_base[:3] = calib["Tr_velo_cam"].reshape(3, 4)

        cam_base2rect = np.eye(4)
        cam_base2rect[:3, :3] = calib["R_rect"].reshape(3, 3)

        P2 = calib["P2:"].reshape(3, 4)
        K_inv = np.linalg.inv(P2[:, :3])
        R_t = P2[:, 3]
        rect2P2 = np.eye(4)
        rect2P2[:3, 3] = np.dot(K_inv, R_t)
        velo2P2 = rect2P2 @ cam_base2rect @ velo2cam_base
        P22imu = np.linalg.inv(rect2P2 @ cam_base2rect @ velo2cam_base @ imu2velo)

        P3 = calib["P3:"].reshape(3, 4)
        K_inv = np.linalg.inv(P3[:, :3])
        R_t = P3[:, 3]
        rect2P3 = np.eye(4)
        rect2P3[:3, 3] = np.dot(K_inv, R_t)
        velo2P3 = rect2P3 @ cam_base2rect @ velo2cam_base
        P32imu = np.linalg.inv(rect2P3 @ cam_base2rect @ velo2cam_base @ imu2velo)

        frames = []
        images: List[ImageInfo] = []
        global_pcs, global_colors = [], []
        lines = open("{}/oxts/{}.txt".format(self.data, self.sequence), "r").readlines()
        im_hws = [
            Image.open("{0}/image_0{1}/{2}/{3:06d}.png".format(self.data, camera, self.sequence, 0))
            for camera in ["2", "3"]
        ]
        im_hws = [(im.height, im.width) for im in im_hws]
        first_imu_pose = None
        for frame, line in enumerate(tqdm(lines)):
            if not frame_in_range(frame, frame_range):
                continue

            frames.append(frame)
            imu_pose = self.read_oxt(line.strip().split())
            if first_imu_pose is None:
                first_imu_pose = imu_pose

            imu_pose = np.linalg.inv(first_imu_pose) @ imu_pose

            # load lidar pc
            lidar_pc = np.fromfile(
                "{0}/velodyne/{1}/{2:06d}.bin".format(self.data, self.sequence, frame), dtype=np.float32
            ).reshape((-1, 4))[:, :3]

            # project to P2 / P3
            uv_2, points_in_2 = self.points_in_image(lidar_pc, velo2P2, P2[:3, :3], im_hws[0])
            uv_3, points_in_3 = self.points_in_image(lidar_pc, velo2P3, P3[:3, :3], im_hws[1])

            # filter with P2 / P3 cam views, distance
            point_mask = np.logical_and(
                np.logical_or(points_in_2, points_in_3), np.linalg.norm(lidar_pc, axis=-1) < self.max_distance
            )
            global_pc = lidar_pc[point_mask]
            uv_2, uv_3 = uv_2[point_mask], uv_3[point_mask]
            points_in_2, points_in_3 = points_in_2[point_mask], points_in_3[point_mask]

            # get colors
            im_2 = np.array(Image.open("{0}/image_0{1}/{2}/{3:06d}.png".format(self.data, "2", self.sequence, frame)))
            im_3 = np.array(Image.open("{0}/image_0{1}/{2}/{3:06d}.png".format(self.data, "3", self.sequence, frame)))
            uv_2 = uv_2[points_in_2].round().astype(int)
            global_col = np.zeros((global_pc.shape[0], 3), dtype=int)
            global_col[points_in_2] = im_2[uv_2[:, 1], uv_2[:, 0]]
            points_only_in3 = np.logical_and(~points_in_2, points_in_3)
            uv_3 = uv_3[points_only_in3].round().astype(int)
            global_col[points_only_in3] = im_3[uv_3[:, 1], uv_3[:, 0]]
            global_colors.append(global_col)

            # transform to global
            global_pc = ((self.cart2hom(global_pc) @ np.linalg.inv(imu2velo).T) @ imu_pose.T)[:, :3]
            global_pcs.append(global_pc)

            for cam_id, camera, transformation, intrinsics in [(0, "2", P22imu, P2), (1, "3", P32imu, P3)]:
                c2w = (imu_pose @ transformation) @ opencv_to_opengl

                image_path = "{0}/image_0{1}/{2}/{3:06d}.png".format(self.data, camera, self.sequence, frame)
                # we store only depth for left camera, because for the right one there are strong occlusion artifacts
                depth_path = (
                    "{0}/depth_0{1}/{2}/{3:06d}.parquet".format(self.data, camera, self.sequence, frame)
                    if cam_id == 0
                    else None
                )

                if depth_path is not None and not os.path.exists(depth_path):
                    lidar2cam = velo2P2 if cam_id == 0 else velo2P3
                    cam_pc = (self.cart2hom(lidar_pc) @ lidar2cam.T)[:, :3]
                    self.save_depth_map(depth_path, cam_pc, intrinsics[:3, :3], im_hws[cam_id])

                img_info = ImageInfo(
                    image_path=image_path,
                    width=im_hws[cam_id][1],
                    height=im_hws[cam_id][0],
                    pose=c2w,
                    intrinsics=intrinsics[:3, :3],
                    depth_path=depth_path,
                    cam_id=cam_id,
                    sequence_id=0,
                    frame_id=frame,
                    timestamp=None,
                    mask_path=None,
                    flow_neighbors=None,
                )
                images.append(img_info)

        annotations = self.get_annotations(
            "{0}/label_02/{1}.txt".format(self.data, self.sequence),
            {im.frame_id: im.pose for im in images if im.cam_id == 0},
            self.max_distance,
        )

        # save pointclouds to disk
        os.makedirs(self.data / f"pointclouds/{self.sequence}", exist_ok=True)
        pointclouds = []
        for frame_id, pc, color in zip(frames, global_pcs, global_colors):
            save_path = str(self.data / f"pointclouds/{self.sequence}/{frame_id:06d}.ply")
            to_ply(save_path, pc, color)
            pointclouds.append(PointCloudInfo(points_path=save_path, sequence_id=0, frame_id=frame_id, timestamp=None))

        global_pcs = np.concatenate(global_pcs)
        bounds = np.array([np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf])
        poses = np.stack([im.pose for im in images])
        pc_min, pc_max = np.min(np.concatenate([global_pcs, poses[:, :3, 3]]), axis=0), np.max(
            np.concatenate([global_pcs, poses[:, :3, 3]]), axis=0
        )
        bounds[:3] = np.minimum(bounds[:3], pc_min)
        bounds[3:] = np.maximum(bounds[3:], pc_max)

        # add 3m vertical offset because LiDAR in KITTI covers only small vertical range
        bounds[5] += 3.0

        assert set(im.frame_id for im in images) == set(
            ann.frame_id for ann in annotations
        ), "Missing annotations for a frame"

        depth_unit_scale_factor = 1.0
        metadata_path = f"metadata_{self.sequence}.pkl" if not self.mars_split else f"metadata_{self.sequence}_mars.pkl"
        metadata_path = self.data / metadata_path
        with open(metadata_path, "wb") as f:
            pickle.dump((images, annotations, bounds, pointclouds, depth_unit_scale_factor), f)

        CONSOLE.log(f"Saved metadata to {metadata_path}", style="bold green")

    def rot_axis(self, angle, axis):
        cg = np.cos(angle)
        sg = np.sin(angle)
        if axis == 0:  # X
            v = [0, 4, 5, 7, 8]
        elif axis == 1:  # Y
            v = [4, 0, 6, 2, 8]
        else:  # Z
            v = [8, 0, 1, 3, 4]
        RX = np.zeros(9, dtype=np.float64)
        RX[v[0]] = 1.0
        RX[v[1]] = cg
        RX[v[2]] = -sg
        RX[v[3]] = sg
        RX[v[4]] = cg
        return RX.reshape(3, 3)

    def rotate(self, vector, angle):
        gamma, beta, alpha = angle[0], angle[1], angle[2]

        # Rotation matrices around the X (gamma), Y (beta), and Z (alpha) axis
        RX = self.rot_axis(gamma, 0)
        RY = self.rot_axis(beta, 1)
        RZ = self.rot_axis(alpha, 2)
        return np.dot(np.dot(np.dot(RZ, RY), RX), vector)

    def read_oxt(self, fields):
        fields = [float(f) for f in fields]
        latlon = fields[:2]
        location = utm.from_latlon(*latlon)
        t = np.array([location[0], location[1], fields[2]])

        roll = fields[3]
        pitch = fields[4]
        yaw = fields[5]
        rotation = self.rotate(np.eye(3), np.array([roll, pitch, yaw]))
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = t
        return T

    def cart2hom(self, pts_3d):
        """Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by appending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def points_in_image(self, points, extrinsics, intrinsics, im_hw):
        cam_pc = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
        uv = self.project_lidar_to_image(cam_pc, intrinsics)
        depths = uv[:, 2]
        mask = np.ones(cam_pc.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, uv[:, 0] >= 0.0)
        mask = np.logical_and(mask, uv[:, 0] < im_hw[1] - 1.0)
        mask = np.logical_and(mask, uv[:, 1] >= 0.0)
        mask = np.logical_and(mask, uv[:, 1] < im_hw[0] - 1.0)
        return uv[:, :2], mask

    def project_lidar_to_image(self, lidar, camera_matrix):
        """Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(lidar)
        campad = np.eye(4)
        campad[: camera_matrix.shape[0], : camera_matrix.shape[1]] = camera_matrix
        pts_2d = np.dot(pts_3d_rect, np.transpose(campad))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d

    def save_depth_map(self, depth_fpath: str, cam_pc: np.ndarray, intrinsics: np.ndarray, im_hw: Tuple[int, int]):
        depth_map = np.zeros(im_hw)
        uv = self.project_lidar_to_image(cam_pc, intrinsics)
        depths = uv[:, 2]
        mask = np.ones(cam_pc.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, uv[:, 0] >= 0.0)
        mask = np.logical_and(mask, uv[:, 0] < im_hw[1] - 1.0)
        mask = np.logical_and(mask, uv[:, 1] >= 0.0)
        mask = np.logical_and(mask, uv[:, 1] < im_hw[0] - 1.0)
        uv = uv[mask]
        depths = depths[mask]
        depth_map[np.floor(uv[:, 1]).astype(np.int32), np.ceil(uv[:, 0]).astype(np.int32)] = depths
        depth_map[np.ceil(uv[:, 1]).astype(np.int32), np.ceil(uv[:, 0]).astype(np.int32)] = depths
        depth_map[np.floor(uv[:, 1]).astype(np.int32), np.floor(uv[:, 0]).astype(np.int32)] = depths
        depth_map[np.ceil(uv[:, 1]).astype(np.int32), np.floor(uv[:, 0]).astype(np.int32)] = depths

        os.makedirs(os.path.dirname(depth_fpath), exist_ok=True)
        # use parquet for compression
        pq.write_table(
            pa.table({"depth": depth_map.flatten()}, metadata={"shape": " ".join([str(x) for x in depth_map.shape])}),
            depth_fpath,
            filesystem=None,
            compression="BROTLI",
        )

    def get_annotations(self, annotation_file: Path, c2ws: Dict[int, np.ndarray], max_distance=80.0):
        with open(annotation_file, "r") as f:
            lines = f.readlines()

        boxes_per_frame = defaultdict(list)
        track_ids = set()
        for line in lines:
            label = line.split(" ")
            if label[2] not in ["Car", "Van", "Bus", "Truck"]:
                continue

            frame_id = int(label[0])
            if frame_id not in c2ws:
                continue
            track_id = int(label[1])
            track_ids.add(track_id)

            h = float(label[10])
            w = float(label[11])
            l = float(label[12])  # noqa: E741
            pos = [float(label[13]), float(label[14]) - h / 2.0, float(label[15])]
            dis_to_cam = np.linalg.norm(pos)
            if dis_to_cam >= max_distance:
                continue
            ry = float(label[16])
            boxes_per_frame[frame_id].append([pos[0], pos[1], pos[2], w, l, h, ry, track_id])

        track_ids = sorted(list(track_ids))
        anns = []
        for frame_id, c2w in c2ws.items():
            c2w = c2w @ opencv_to_opengl

            if frame_id in boxes_per_frame:
                boxes = np.array(boxes_per_frame[frame_id], dtype=np.float32)

                # cam to global
                boxes[:, :3] = (c2w[None, :3, :3] @ boxes[:, :3, None]).squeeze(-1) + c2w[:3, 3]
                for i, box in enumerate(boxes):
                    rotation_matrix = np.array(
                        [[np.cos(box[6]), 0.0, np.sin(box[6])], [0.0, 1.0, 0.0], [-np.sin(box[6]), 0.0, np.cos(box[6])]]
                    )
                    # cam to imu: xyz (right,down,front) -> zxy (left,front,up)
                    rotation_matrix = c2w[:3, :3] @ rotation_matrix
                    point = rotation_matrix @ np.array([[1, 0, 0]]).T
                    boxes[i, 6] = np.arctan2(-point[0], point[1])

                box_ids = np.array([track_ids.index(int(box[7])) for box in boxes], dtype=np.int64)
                ann = AnnotationInfo(
                    sequence_id=0,
                    frame_id=frame_id,
                    timestamp=None,
                    boxes=boxes[:, :7],
                    box_ids=box_ids,
                    class_ids=np.zeros(len(boxes), dtype=np.int64),
                )
            else:
                ann = AnnotationInfo(
                    sequence_id=0,
                    frame_id=frame_id,
                    timestamp=None,
                    boxes=np.empty((0, 7)),
                    box_ids=np.empty((0,), dtype=np.int64),
                    class_ids=np.empty((0,), dtype=np.int64),
                )
            anns.append(ann)

        return anns


VKITTI2_GROUND_PLANE_Z = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

VKITTI2_FRAME_RANGE = {"02": [100, 200], "06": [0, 100], "18": [273, 338]}


@dataclass
class ProcessVKITTI2:
    data: Path = Path("data/VKITTI2/")
    """path to the data"""
    sequence: Literal["02", "06", "18"] = "02"
    """sequence to use"""
    max_distance: float = 200.0
    """The maximum distance to use for filtering the pointclouds and 3D boxes. If 0, no max distance."""

    def main(self):
        data_dir = self.data / f"Scene{self.sequence}" / "clone"

        images: list[ImageInfo] = []
        annotations: list[AnnotationInfo] = []
        with open(data_dir / "intrinsic.txt", "r") as in_f, open(data_dir / "extrinsic.txt", "r") as ex_f:
            # frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
            next(in_f)
            # frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
            next(ex_f)

            for in_line, ex_line in tqdm(zip(in_f, ex_f)):
                in_entry = in_line.strip().split()
                frame = int(in_entry[0])
                if not frame_in_range(frame, VKITTI2_FRAME_RANGE[self.sequence]):
                    continue

                camera_id = int(in_entry[1])
                fx, fy, cx, cy = [float(x) for x in in_line.strip().split()[2:]]
                intrinsics = np.eye(3)
                intrinsics[0, 0] = fx
                intrinsics[1, 1] = fy
                intrinsics[0, 2] = cx
                intrinsics[1, 2] = cy

                w2c = np.array([float(x) for x in ex_line.strip().split()[2:]]).reshape(4, 4)
                c2w = VKITTI2_GROUND_PLANE_Z @ (np.linalg.inv(w2c) @ opencv_to_opengl)

                image_path = "{0}/frames/rgb/Camera_{1}/rgb_{2:05d}.jpg".format(data_dir, camera_id, frame)
                depth_path = "{0}/frames/depth/Camera_{1}/depth_{2:05d}.png".format(data_dir, camera_id, frame)
                image = Image.open(image_path)

                item = ImageInfo(
                    image_path=image_path,
                    width=image.width,
                    height=image.height,
                    pose=c2w,
                    intrinsics=intrinsics,
                    mask_path=None,
                    depth_path=depth_path,
                    sequence_id=0,
                    cam_id=camera_id,
                    frame_id=frame,
                    timestamp=None,
                    flow_neighbors=None,
                )
                images.append(item)

        annotations, pointclouds, bounds = self.get_annotations_pointclouds_bounds(
            data_dir / "pose.txt", images, VKITTI2_FRAME_RANGE[self.sequence]
        )

        depth_unit_scale_factor = 1 / 100
        with open(self.data / f"metadata_{self.sequence}.pkl", "wb") as f:
            pickle.dump((images, annotations, bounds, pointclouds, depth_unit_scale_factor), f)
        CONSOLE.log(f"Saved metadata to {self.data / f'metadata_{self.sequence}.pkl'}")

    def get_annotations_pointclouds_bounds(self, annotation_file: Path, images: list[ImageInfo], frame_range):
        track_ids = set()
        boxes_per_frame = defaultdict(list)

        # load annotations
        with open(annotation_file, "r") as bbox_f:
            # frame objectID left top right bottom
            next(bbox_f)
            for bbox_line in bbox_f:
                bbox_entry = bbox_line.strip().split()
                frame = int(bbox_entry[0])
                if not frame_in_range(frame, frame_range):
                    continue

                center = np.array([float(x) for x in bbox_entry[7:10]]) @ VKITTI2_GROUND_PLANE_Z[:3, :3].T
                w, h, l = [float(x) for x in bbox_entry[4:7]]  # noqa: E741
                center[2] = center[2] + h / 2
                ry = -float(bbox_entry[10]) - np.pi / 2
                box = np.array([*center, w, l, h, ry])

                # filter out duplicate boxes
                if any([all(b[:7] == box) for b in boxes_per_frame[frame]]):
                    continue

                object_id = int(bbox_entry[2])
                track_ids.add(object_id)
                boxes_per_frame[frame].append([*box, object_id])

        frames = sorted(list(set([im.frame_id for im in images])))
        global_pcs, global_colors = [], []
        annotations = []
        track_ids = sorted(list(track_ids))
        for frame_id in tqdm(frames):
            frame_pc, frame_colors = [], []
            # create pointcloud from depths
            for im in images:
                if im.frame_id == frame_id:
                    intr, pose = torch.from_numpy(im.intrinsics).float(), torch.from_numpy(im.pose).float()
                    depth = get_depth_image_from_path(
                        filepath=Path(im.depth_path), height=im.height, width=im.width, scale_factor=1 / 100
                    )
                    colors = torch.from_numpy(np.array(Image.open(im.image_path))).view(-1, 3).float()
                    points = depth_to_points(depth.squeeze(-1), intr)
                    colors = colors[points[:, 2] > 0]
                    points = points[points[:, 2] > 0]
                    colors = colors[points[:, 2] < self.max_distance]
                    points = points[points[:, 2] < self.max_distance]
                    points = points @ opencv_to_opengl[:3, :3]
                    points = transform_points(points, pose)
                    frame_pc.append(points.cpu().numpy())
                    frame_colors.append(colors.cpu().numpy())

            global_pcs.append(np.concatenate(frame_pc, axis=0))
            global_colors.append(np.concatenate(frame_colors, axis=0))

            if frame_id in boxes_per_frame:
                boxes = np.array(boxes_per_frame[frame_id], dtype=np.float32)
                box_ids = np.array([track_ids.index(int(box[7])) for box in boxes])

                # hack to fix incorrect truck box in sequence 06
                if self.sequence == "06":
                    boxes[box_ids == 9, 3] += 0.5

                ann = AnnotationInfo(
                    sequence_id=0,
                    frame_id=frame_id,
                    timestamp=None,
                    boxes=boxes[:, :7],
                    box_ids=box_ids,
                    class_ids=np.zeros(len(boxes)),
                )
            else:
                ann = AnnotationInfo(
                    sequence_id=0,
                    frame_id=frame_id,
                    timestamp=None,
                    boxes=np.empty((0, 7)),
                    box_ids=np.empty((0,)),
                    class_ids=np.empty((0,)),
                )
            annotations.append(ann)

        bounds = np.array([np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf])
        poses = np.stack([im.pose for im in images])
        cat_pc = np.concatenate([np.concatenate(global_pcs, axis=0), poses[:, :3, 3]], axis=0)
        pc_min, pc_max = np.min(cat_pc, axis=0), np.max(cat_pc, axis=0)
        bounds[:3] = np.minimum(bounds[:3], pc_min)
        bounds[3:] = np.maximum(bounds[3:], pc_max)

        # save pointclouds to disk
        os.makedirs(self.data / f"Scene{self.sequence}/pointclouds", exist_ok=True)
        pointclouds = []
        for frame_id, pc, color in zip(frames, global_pcs, global_colors):
            save_path = str(self.data / f"Scene{self.sequence}/pointclouds/{frame_id:06d}.ply")
            to_ply(save_path, pc, color)
            pointclouds.append(PointCloudInfo(points_path=save_path, sequence_id=0, frame_id=frame_id, timestamp=None))

        return annotations, pointclouds, bounds


Commands = Union[
    Annotated[ProcessArgoverse2, tyro.conf.subcommand(name="av2")],
    Annotated[ProcessKITTI, tyro.conf.subcommand(name="kitti")],
    Annotated[ProcessVKITTI2, tyro.conf.subcommand(name="vkitti2")],
    Annotated[ProcessWaymo, tyro.conf.subcommand(name="waymo")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    try:
        tyro.cli(Commands).main()
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + e.args[0])


if __name__ == "__main__":
    entrypoint()
