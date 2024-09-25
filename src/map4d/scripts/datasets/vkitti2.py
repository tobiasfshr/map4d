import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from tqdm import tqdm

from map4d.common.geometry import depth_to_points, opencv_to_opengl
from map4d.common.io import get_depth_image_from_path, to_ply
from map4d.common.pointcloud import transform_points
from map4d.data.parser.typing import AnnotationInfo, ImageInfo, PointCloudInfo
from map4d.scripts.datasets.util import frame_in_range

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
