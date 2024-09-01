"""Depth dataset suppoting parquet files."""

from typing import Dict

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from torch.utils.data import Dataset

from map4d.common.io import get_depth_image_from_path


class DepthDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "depth_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["depth_filenames"] is not None
        )
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

    def get_metadata(self, data: Dict) -> Dict:
        if self.depth_filenames is None:
            return {"depth_image": self.depths[data["image_idx"]]}

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )
        return {"depth_image": depth_image}


class CameraImageDataset(Dataset):
    """Torch Dataset that returns images and cameras."""

    def __init__(self, dataset: InputDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.dataset[image_idx]
        image_idx = data["image_idx"]
        camera = self.dataset.cameras[image_idx : image_idx + 1]
        if camera.metadata is None:
            camera.metadata = {}
        assert len(self.dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return camera, data
