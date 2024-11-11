
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
Use our preprocessing scripts to prepare the datasets:
```
mp-process [av2|kitti|vkitti2|waymo]
```

We provided detailed instructions for preparing the supported datasets in our documentation: 
- [Argoverse 2](docs/datasets/Argoverse2.md)
- [KITTI](docs/datasets/KITTI.md)
- [VKITTI2](docs/datasets/VKITTI2.md)
- [Waymo Open](docs/datasets//Waymo.md)

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
@InProceedings{fischer2024dynamic,
    author    = {Tobias Fischer and Jonas Kulhanek and Samuel Rota Bul{\`o} and Lorenzo Porzi and Marc Pollefeys and Peter Kontschieder},
    title     = {Dynamic 3D Gaussian Fields for Urban Areas},
    booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year      = {2024}
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
