import json
import os
import pickle
import shutil
import subprocess
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


# Residential split
RESIDENTIAL_SEQUENCE_IDS = [
    "0c61aea3-3cba-35f3-8971-df42cd5b9b1a",
    "7c30c3fc-ea17-38d8-9c52-c75ccb112253",
    "a2f568b5-060f-33f0-9175-7e2062d86b6c",
    "b9f73e2a-292a-3876-b363-3ebb94584c7a",
    "cea5f5c2-e786-30f5-8305-baead8923063",
    "6b0cc3b0-2802-33a7-b885-f1f1409345ac",
    "7cb4b11f-3872-3825-83b5-622e1a2cdb28",
    "a359e053-a350-36cf-ab1d-a7980afaffa2",
    "c654b457-11d4-393c-a638-188855c8f2e5",
    "f41d0e8f-856e-3f7d-a3f9-ff5ba7c8e06d",
    "6f2f7d1e-8ded-35c5-ba83-3ca906b05127",
    "8aad8778-73ce-3fa0-93c7-804ac998667d",
    "b51561d9-08b0-3599-bc78-016f1441bb91",
    "c990cafc-f96c-3107-b213-01d217b11272",
]

# Downtown split
DOWNTOWN_SEQUENCE_IDS = [
    "05853f69-f948-3d04-8d64-d4e721c0e1a5",
    "05fb81ab-5e46-3f63-a59f-82fc66d5a477",
    "150ae964-5091-3681-b738-88715052c792",
    "1a6487dd-8bc6-3762-bd8a-2e50e15dbe75",
    "37fcd8ac-c148-3c4a-92ac-a10f355451b7",
    "422dd53b-6010-4eb9-8902-de3d134c5a70",
    "51e6b881-e5a1-30c8-ae2b-02891d5a54ce",
    "5bc5e7b0-4d90-3ac8-8ca8-f6037e1cf75c",
    "5d9c1080-e6e9-3222-96a2-37ca7286a874",
    "6bae6c0c-5296-376d-96bc-6c8dbe6693a5",
    "6e106cf8-f6dd-38f6-89c8-9be7a71e7275",
    "8184872e-4203-3ff1-b716-af5fad9233ec",
    "8606d399-57d4-3ae9-938b-db7b8fb7ef8c",
    "89f79c55-6698-3037-bd2e-d40c81af169a",
    "9158b740-6527-3194-9953-6b7b3b28d544",
    "931b76ee-63df-36f6-9f2e-7fb16f2ee721",
    "9eb87a0b-2457-359d-b958-81e8583d8e44",
    "9efe1171-6faf-3427-8451-8f6469f7678e",
    "bd9636d2-7220-3585-9c7d-4acaea167b71",
    "c8cdffb0-7942-3ff5-9f71-210e095e1d31",
    "d0828f48-3e67-3136-9c70-1f99968c8280",
    "e453f164-dd36-3f1a-9471-05c2627cbaa5",
    "fb720691-1736-3fa2-940b-07b97603efc6",
]


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

    def _download_data(self, split: str):
        split_name = "train" if split in ["train", "residential", "downtown"] else split
        CONSOLE.log(f"Downloading Argoverse2 {split} split...")
        target_dir = self.data / split_name

        # Create an S3 client with no signature required
        if split == "residential":
            sequences = RESIDENTIAL_SEQUENCE_IDS
            for sequence in sequences:
                s3_path = f"s3://argoverse/datasets/av2/sensor/{split_name}/{sequence}/*"
                self._download_s3(s3_path, target_dir / sequence)
        elif split == "downtown":
            sequences = DOWNTOWN_SEQUENCE_IDS
            for sequence in sequences:
                s3_path = f"s3://argoverse/datasets/av2/sensor/{split_name}/{sequence}/*"
                self._download_s3(s3_path, target_dir / sequence)
        else:
            s3_path = f"s3://argoverse/datasets/av2/sensor/{split_name}/*"
            self._download_s3(s3_path, target_dir)

    def _download_s3(self, s3_path: str, target_dir: Path):
        assert shutil.which(
            "s5cmd"
        ), "s5cmd is not installed. Please install it with e.g. 'conda install s5cmd -c conda-forge'."
        os.makedirs(target_dir, exist_ok=True)
        command = ["s5cmd", "--no-sign-request", "cp", s3_path, str(target_dir)]
        CONSOLE.log(f"Downloading {s3_path} to {str(target_dir)}")
        subprocess.run(command, check=True)

    def _check_exists(self, sequences: list[str]):
        for seq in sequences:
            if not (self.data / self.split / seq).exists():
                return False
        return True

    def _check_data(self):
        if self.location_aabb is not None:
            if self.city == "PIT" and self.location_aabb == (6180, 1620, 6310, 1780):
                # residential split
                if not self._check_exists(RESIDENTIAL_SEQUENCE_IDS):
                    self._download_data("residential")
            elif self.city == "PIT" and self.location_aabb == (1100, -50, 1220, 150):
                # downtown split
                if not self._check_exists(DOWNTOWN_SEQUENCE_IDS):
                    self._download_data("downtown")
            else:
                # any other split defined by location AABB
                self._download_data(self.split)
        else:
            # if no location AABB is set, we need a log id in a given split
            assert self.log_id is not None
            if not self._check_exists([self.log_id]):
                s3_path = f"s3://argoverse/datasets/av2/sensor/{self.split}/{self.log_id}/*"
                self._download_s3(s3_path, self.data / self.split / self.log_id)

    def main(self):
        if av2 is None:
            CONSOLE.log(
                "AV2 API is not installed. Please install it with `pip install git+https://github.com/tobiasfshr/av2-api.git`.",
                style="bold red",
            )
            return
        self._check_data()
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
