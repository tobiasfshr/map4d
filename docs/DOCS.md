# Documentation

Here we provide details about data conventions and alike to make the code easier to understand, use and extend.

## Coordinate system conventions

We process all datasets into an intermediate format, which shares the coordinate frame definitions with nerfstudio. In particular, use the OpenGL/Blender (and original NeRF) coordinate convention for cameras. +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction. The world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane. All poses are in camera-to-world format.

The 4 DoF object poses are provided in the world coordinate system, parameterized by a 3D center position and a yaw angle in radians (rotation around the Z axis). The object dimensions are provided in `[width, length, height]` order where in the neutral position (i.e. yaw angle 0) the box points in +Y direction.

## Time representation

We represent time as a float [-1, 1] in data and model.

## Intermediate data format for the dataparser

We parse the intermediate format with a common dataparser (map4d.data.parser.base.StreetSceneParser). The data should be sorted by:
1. Sequences
2. Frames
3. Cameras

This is necessary for the pose optimization, which assumes a certain ordering of the poses to group them according to cameras.

Example:
```
images = [
    ImageInfo(sequence 0 frame 0 camera 0)
    ImageInfo(sequence 0 frame 0 camera 1)
    ImageInfo(sequence 0 frame 1 camera 0)
    ImageInfo(sequence 0 frame 1 camera 1)
    ...

    ImageInfo(sequence 1 frame 0 camera 0)
    ImageInfo(sequence 1 frame 0 camera 1)
    ImageInfo(sequence 0 frame 1 camera 0)
    ImageInfo(sequence 0 frame 1 camera 1)
    ...
]
```
The annotations are provided per-frame and further are indepedent from train / eval splits, provided as a single list to the model.


## Metrics
Due to a memory leak in the torchmetrics package, we use alternative implementations. We make sure they produce the same results (see tests/metric_test.py). Note that, when computing per-pixel ssim, the results can be *slightly* different due to padding. The deviation is usually within +/- 0.002 in evaluation.
