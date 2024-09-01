from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from plyfile import PlyData, PlyElement
from torch import Tensor


def from_ply(filename: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load point cloud from PLY file.

    Args:
        filename (str): Path to PLY file.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Point cloud and color array.
    """
    # Load the PLY file
    ply_data = PlyData.read(filename)

    # Extract X, Y, and Z coordinates into a NumPy array
    x_coordinates = ply_data["vertex"]["x"]
    y_coordinates = ply_data["vertex"]["y"]
    z_coordinates = ply_data["vertex"]["z"]

    # Extract R, G, and B values into a NumPy array
    if "red" in ply_data["vertex"]:
        assert (
            "green" in ply_data["vertex"] and "blue" in ply_data["vertex"]
        ), "Color channels must be present together!"
        red = ply_data["vertex"]["red"]
        green = ply_data["vertex"]["green"]
        blue = ply_data["vertex"]["blue"]
        color_array = np.column_stack((red, green, blue))
    else:
        color_array = None

    # Combine X, Y, and Z coordinates, colors into a Nx3 NumPy array
    point_cloud_array = np.column_stack((x_coordinates, y_coordinates, z_coordinates))
    return point_cloud_array, color_array


def to_ply(
    filename: str,
    points: Tensor | np.ndarray,
    colors: Tensor | np.ndarray | None = None,
    faces: Tensor | np.ndarray | None = None,
):
    """N,3 points / colors to ply file."""
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if colors is not None and not isinstance(colors, np.ndarray):
        colors = colors.cpu().numpy()
    if faces is not None and not isinstance(faces, np.ndarray):
        faces = faces.cpu().numpy()

    vertexs = np.array(
        [tuple(v) for v in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    if colors is not None:
        vertex_colors = np.array(
            [tuple(v) for v in colors.astype(np.uint8)],
            dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
        for prop in vertexs.dtype.names:
            vertex_all[prop] = vertexs[prop]
        for prop in vertex_colors.dtype.names:
            vertex_all[prop] = vertex_colors[prop]
        vertexs = vertex_all

    data = [PlyElement.describe(vertexs, "vertex")]
    if faces is not None:
        faces_container = np.empty(len(faces), dtype=[("vertex_indices", "i4", (faces.shape[-1],))])
        faces_container["vertex_indices"] = faces
        data.append(PlyElement.describe(faces_container, "face"))
    PlyData(data).write(filename)


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.

    Filepath points to a 16-bit or 32-bit depth image, a numpy array `*.npy` or a parquet file `*.parquet`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath is None:
        return torch.zeros((height, width, 1), dtype=torch.float32)

    if filepath.suffix == ".npy":
        image = np.load(filepath)
    elif filepath.suffix == ".parquet":
        table = pq.read_table(filepath)
        size = [int(x) for x in table.schema.metadata[b"shape"].split()]
        image = table["depth"].to_numpy().reshape(*size)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        # VKITTI2 hack
        image[image == 65535] = 0

    image = image.astype(np.float64) * scale_factor
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis]).float()
