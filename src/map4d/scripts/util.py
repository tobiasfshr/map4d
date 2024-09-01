import numpy as np


def tile_cameras(named_sensors: dict[str, np.ndarray]) -> np.ndarray:
    """Combine ring cameras into a tiled image.

    Layout:

        ##########################################################
        # ring_front_left # ring_front_center # ring_front_right #
        ##########################################################
        # ring_side_left  #                   #  ring_side_right #
        ##########################################################
        ############ ring_rear_left # ring_rear_right ############
        ##########################################################

    Args:
        named_sensors: Dictionary of camera names to image paths.

    Returns:
        Tiled image.
    """
    landscape_width = 2048
    landscape_height = 1550

    height = landscape_height + landscape_height + landscape_height
    width = landscape_width + landscape_height + landscape_width
    tiled_im = np.zeros((height, width, 3), dtype=np.uint8)

    if "ring_front_left" in named_sensors:
        ring_front_left = named_sensors["ring_front_left"]
        tiled_im[:landscape_height, :landscape_width] = ring_front_left

    if "ring_front_center" in named_sensors:
        ring_front_center = named_sensors["ring_front_center"]
        tiled_im[:landscape_width, landscape_width : landscape_width + landscape_height] = ring_front_center

    if "ring_front_right" in named_sensors:
        ring_front_right = named_sensors["ring_front_right"]
        tiled_im[:landscape_height, landscape_width + landscape_height :] = ring_front_right

    if "ring_side_left" in named_sensors:
        ring_side_left = named_sensors["ring_side_left"]
        tiled_im[landscape_height : 2 * landscape_height, :landscape_width] = ring_side_left

    if "ring_side_right" in named_sensors:
        ring_side_right = named_sensors["ring_side_right"]
        tiled_im[
            landscape_height : 2 * landscape_height,
            landscape_width + landscape_height :,
        ] = ring_side_right

    if "ring_rear_left" in named_sensors:
        ring_rear_left = named_sensors["ring_rear_left"]
        tiled_im[2 * landscape_height : 3 * landscape_height, :landscape_width] = ring_rear_left

    if "ring_rear_right" in named_sensors:
        ring_rear_right = named_sensors["ring_rear_right"]
        tiled_im[2 * landscape_height : 3 * landscape_height, width - landscape_width :] = ring_rear_right
    return tiled_im


def grid_cameras(named_sensors: dict[str, np.ndarray], col_size: int) -> np.ndarray:
    """Aligns a dict of images in a grid.

    Parameters:
        named_sensors: dict of NumPy arrays representing images.
        col_size: int specifying the columns per row.

    Returns:
        A NumPy array representing the aligned grid of images.
    """

    num_images = len(named_sensors)

    if num_images == 0:
        raise ValueError("No images provided.")

    col_size = min(col_size, num_images)

    # Calculate the number of columns based on the row size
    row_size = (num_images + col_size - 1) // col_size

    # Get max image dimensions
    height, width, channels = max([image.shape for image in named_sensors.values()])

    # Create an empty canvas for the grid
    canvas = np.zeros((height * row_size, width * col_size, channels), dtype=np.uint8)

    for idx, (key, image) in enumerate(named_sensors.items()):
        row = idx // col_size
        col = idx % col_size
        # pad image if needed
        padded_image = np.zeros((height, width, channels), dtype=np.uint8)
        padded_image[: image.shape[0], : image.shape[1], :] = image
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width, :] = padded_image

    return canvas
