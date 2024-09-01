"""Optimizer for vehicle poses which ties together poses of cameras of the ego vehicle."""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from torch import Tensor, device


@dataclass
class VehiclePoseOptimizerConfig(CameraOptimizerConfig):
    """Configuration of optimization for vehicle poses."""

    _target: Type = field(default_factory=lambda: VehiclePoseOptimizer)


class VehiclePoseOptimizer(CameraOptimizer):
    """Optimizer for vehicle poses which ties together poses of cameras of the ego vehicle.

    Assumes that the cameras are ordered by vehicle pose, i.e. the first N cameras belong to the
    first vehicle pose, the next N cameras belong to the second vehicle pose, etc.
    """

    def __init__(
        self, config: CameraOptimizerConfig, num_cameras: int, num_physical_cameras: int, device: device | str, **kwargs
    ) -> None:
        self.num_physical_cameras = num_physical_cameras  # N
        num_poses = num_cameras // num_physical_cameras
        assert num_poses * num_physical_cameras == num_cameras
        super().__init__(config, num_poses, device, **kwargs)

    def forward(self, indices: Tensor) -> Tensor:
        vehicle_pose_indices = indices // self.num_physical_cameras
        pose_deltas = super().forward(vehicle_pose_indices)
        return pose_deltas
