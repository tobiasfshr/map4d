import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from map4d.common.geometry import opencv_to_opengl
from map4d.common.io import to_ply
from map4d.data.parser.typing import AnnotationInfo, ImageInfo, PointCloudInfo

try:
    import av2
    import av2.utils.io as io_utils
    from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader, convert_pose_dataframe_to_SE3
    from av2.geometry.se3 import SE3
    from av2.structures.cuboid import CuboidList
except ImportError:
    av2 = None


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
                self.alignment_path / f"{seq_name}_transforms.json"
            ), f"Alignment file {self.alignment_path / f'{seq_name}_transforms.json'} does not exist."
            with open(self.alignment_path / f"{seq_name}_transforms.json", "r") as f:
                transforms_per_seq = json.load(f)
            transforms_per_seq = [np.array(transforms_per_seq[log]) for log in log_ids]
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
