# [Argoverse 2](https://www.argoverse.org/av2.html)

This dataset is a collection of open-source autonomous driving data and high-definition (HD) maps from six U.S. cities: Austin, Detroit, Miami, Pittsburgh, Palo Alto, and Washington, D.C. This release builds upon the initial launch of Argoverse (“Argoverse 1”), which was among the first data releases of its kind to include HD maps for machine learning and computer vision research.

We provide scripts to download and preprocess the parts of the Argoverse 2 dataset used in our experiments. We refer to the [Argoverse User Guide](https://argoverse.github.io/user-guide/getting_started.html#overview) for detailed instructions on how to get started with the dataset.

## Requirements
To download and preprocess the Argoverse 2 dataset:

1. **Install our modified [Argoverse 2 devkit](https://argoverse.github.io/user-guide/getting_started.html) via**
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly-2023-12-11
pip install git+https://github.com/tobiasfshr/av2-api.git
```
We use the following rustc version: `rustc 1.76.0-nightly (21cce21d8 2023-12-11)`.

2. **Install `s5cmd`**

#### Conda Installation (Recommended)

```bash
conda install s5cmd -c conda-forge
```

#### Manual Installation

```bash
#!/usr/bin/env bash

export INSTALL_DIR=$HOME/.local/bin
export PATH=$PATH:$INSTALL_DIR
export S5CMD_URI=https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_$(uname | sed 's/Darwin/macOS/g')-64bit.tar.gz

mkdir -p $INSTALL_DIR
curl -sL $S5CMD_URI | tar -C $INSTALL_DIR -xvzf - s5cmd
```

Note that it will install s5cmd in your local bin directory. You can always change the path if you prefer installing it in another directory. Note that an AWS account is **not** required to download the datasets.

## Download & Preprocessing
For Argoverse 2, we provide ego-vehicle masks located at `assets/masks`.


```
# Residential split
mp-process av2 --location-aabb 6180 1620 6310 1780

# Downtown split
mp-process av2 --location-aabb 1100 -50 1220 150

# Single sequence
mp-process av2
```

By default, this will download and preprocess the dataset in the following folder structure:
```
data/
    Argoverse2/
        train/
            0c61aea3-3cba-35f3-8971-df42cd5b9b1a/
            ...
```
You can adjust the path with the `--data` option.
