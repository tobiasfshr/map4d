from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImageInfo:
    image_path: str
    width: int
    height: int
    pose: np.ndarray  # cam2world
    intrinsics: np.ndarray
    mask_path: str
    depth_path: str
    cam_id: int
    sequence_id: int
    frame_id: int
    timestamp: float
    flow_neighbors: list[FlowNeighbor] | None
    cam2vehicle: Optional[np.ndarray] = None


@dataclass
class FlowNeighbor:
    image: ImageInfo
    flow_path: str | None


@dataclass
class AnnotationInfo:
    sequence_id: int
    frame_id: int
    timestamp: float
    boxes: np.ndarray  # in world coordinate
    box_ids: np.ndarray
    class_ids: np.ndarray  # 0: car / truck / bus, 1: pedestrian / bicyclist
    vehicle2world: Optional[np.ndarray] = None


@dataclass
class PointCloudInfo:
    points_path: str
    sequence_id: int
    frame_id: int
    timestamp: float
    vehicle2world: Optional[np.ndarray] = None  # if None, then points are in world frame
    colors: Optional[np.ndarray] = None
