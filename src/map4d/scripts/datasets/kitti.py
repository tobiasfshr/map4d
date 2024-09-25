import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import utm
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from tqdm import tqdm

from map4d.common.geometry import opencv_to_opengl
from map4d.common.io import to_ply
from map4d.data.parser.typing import AnnotationInfo, ImageInfo, PointCloudInfo
from map4d.scripts.datasets.util import frame_in_range

KITTI_FRAME_RANGE = {"0001": [380, 431], "0002": [140, 224], "0006": [65, 120]}


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
