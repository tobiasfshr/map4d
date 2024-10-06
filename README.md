
<div align="center">

# **Map4D**

![Alt Text](assets/media/teaser.gif)

### A framework for photo-realistic mapping of dynamic urban areas

</div>

## Requirements
This project was tested with the following dependencies
- Python 3.10
- CUDA 11.8
- PyTorch 2.0.1

## Installation
1. Clone the repository
```
git clone https://github.com/tobiasfshr/map4d.git
```

2. Create a new conda environment
```
conda create --name map4d -y python=3.10
conda activate map4d
pip install --upgrade pip
```

3. Install PyTorch, CUDA, tinycudann (fork by [@hturki](https://github.com/hturki)), Nerfstudio and this project
```
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/hturki/tiny-cuda-nn.git@ht/res-grid#subdirectory=bindings/torch
pip install nerfstudio==1.0.3
python setup.py develop
```

## Data
We support the following datasets:
- [Argoverse 2](docs/Argoverse2.md) (Sensor dataset)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) (tracking split)
- [VKITTI2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
- [Waymo Open](https://waymo.com/open/)

Download the datasets to a location of your convenience. You can later adjust the data path in the preprocessing script. Note that we provide a joint download & preprocess utility for Waymo (see below).

By default, we assume the following folder structure:
```
data/
    Argoverse2/
        train/
            0c61aea3-3cba-35f3-8971-df42cd5b9b1a/
            ...

    KITTI/
        tracking/
            training/
                image_02/
                ...

    VKITTI2/
        Scene01/
        ...
    waymo/
        ...
```

For Argoverse 2, the ego-vehicle masks are located at `assets/masks` by default.

Generate the necessary metadata files with:
```
mp-process [av2|kitti|vkitti2|waymo]
```

To prepare the full datasets, run:

### VKITTI2
```
mp-process vkitti2 --sequence 02
mp-process vkitti2 --sequence 06
mp-process vkitti2 --sequence 18
```

### KITTI

```
mp-process kitti --sequence 0001
mp-process kitti --sequence 0002
mp-process kitti --sequence 0006
```
### Waymo
```
mp-process waymo
```

This will download and preprocess the full Dynamic-32 split from [EmerNeRF](https://emernerf.github.io/).

### Argoverse 2
```
# Residential split
mp-process av2 --location-aabb 6180 1620 6310 1780

# Downtown split
mp-process av2 --location-aabb 1100 -50 1220 150

# Single sequence
mp-process av2
```

NOTE: For Argoverse 2, install the modified [devkit](https://argoverse.github.io/user-guide/getting_started.html) via
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly-2023-12-11
pip install git+https://github.com/tobiasfshr/av2-api.git
```
We use the following rustc version: `rustc 1.76.0-nightly (21cce21d8 2023-12-11)`.

## Models

We provide detailed instructions for reproducing the experiments for the supported models:
- [Multi-Level Neural Scene Graphs](docs/models/MLNSG.md)
- [Dynamic 3D Gaussian Fields](docs/models/4DGF.md)

## Training and Evaluation
Use the generated metadata files to train the model on a specific dataset:
```
ns-train <model_name> --data <filepath_to_metadata>
```

The models will be automatically evaluated when training finishes. To evaluate a trained model manually, execute

```
ns-eval --load-config <trained_model_config>
```

## Rendering
Render a single evaluation view with

```
mp-render view --image-idx <some_index> --load-config <path_to_trained_model>
```

To render the reference trajectory, run

```
mp-render camera-path --load-config <path_to_trained_model>
```
You can also use the nerfstudio viewer to generate your own camera paths or use arguments like smooth, rotate and mirror to alter the reference trajectory.

## Viewer
You can visualize your training with the viewer directly via specifying `--vis viewer` or visualize finished trainings with:
```
ns-viewer --load-config <path_to_trained_model>
```
We add sliders for time and sequence ID to control which objects are rendered at which location in the viewer.

## Documentation

For more detailed information on the codebase, please check out our [documentation](docs/DOCS.md).

## Citation
Please consider citing our work with the following references
```
@article{fischer2024dynamic,
  title={Dynamic 3D Gaussian Fields for Urban Areas},
  author={Fischer, Tobias and Kulhanek, Jonas and Bul{\`o}, Samuel Rota and Porzi, Lorenzo and Pollefeys, Marc and Kontschieder, Peter},
  journal={arXiv preprint arXiv:2406.03175},
  year={2024}
}

@InProceedings{fischer2024multi,
    author    = {Fischer, Tobias and Porzi, Lorenzo and Rota Bul\`{o}, Samuel and Pollefeys, Marc and Kontschieder, Peter},
    title     = {Multi-Level Neural Scene Graphs for Dynamic Urban Environments},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024}
}
```
## Acknowledgements

This project builds on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). Parts of the data processing code and the video embedding implementation are based on [SUDS](https://github.com/hturki/suds). The Waymo preprocessing code is based on [EmerNeRF](https://github.com/NVlabs/EmerNeRF/).
