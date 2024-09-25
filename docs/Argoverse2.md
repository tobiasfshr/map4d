# Downloading Argoverse 2

[Argoverse 2](https://www.argoverse.org/av2.html) is a collection of open-source autonomous driving data and high-definition (HD) maps from six U.S. cities: Austin, Detroit, Miami, Pittsburgh, Palo Alto, and Washington, D.C. This release builds upon the initial launch of Argoverse (“Argoverse 1”), which was among the first data releases of its kind to include HD maps for machine learning and computer vision research.


We use Argoverse 2 Sensor Dataset for [4DGF](models/4DGF.md). Argoverse 2 Sensor Dataset contains 1,000 3D annotated scenarios with lidar, stereo imagery, and ring camera imagery. This dataset improves upon the Argoverse 1 3D Tracking dataset.

This file is a download guidebook for the part of Argoverse dataset used in 4DGF. We refer to the [Argoverse User Guide](https://argoverse.github.io/user-guide/getting_started.html#overview) for detailed instructions on how to get started with the dataset.

## Setup

We *highly* recommend using `conda` with the `conda-forge` channel for package management.

Argoverse 2 datasets are available for download from AWS S3.

For the best experience, we highly recommend using the open-source [`s5cmd`](https://github.com/peak/s5cmd) tool to transfer the data to your local filesystem. Please note that an AWS account is **not** required to download the datasets.

## Install `s5cmd`

### Conda INstallation (Recommended)

```bash
conda install s5cmd -c conda-forge
```

### Manual Installation

```bash
#!/usr/bin/env bash

export INSTALL_DIR=$HOME/.local/bin
export PATH=$PATH:$INSTALL_DIR
export S5CMD_URI=https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_$(uname | sed 's/Darwin/macOS/g')-64bit.tar.gz

mkdir -p $INSTALL_DIR
curl -sL $S5CMD_URI | tar -C $INSTALL_DIR -xvzf - s5cmd
```

Note that it will install s5cmd in your local bin directory. You can always change the path if you prefer installing it in another directory.


## Download the Dataset

### Residential Split

```bash
#!/usr/bin/env bash

# Dataset URIs
# s3://argoverse/datasets/av2/sensor/

# Sequence IDs to download
SEQUENCE_IDS=(
  0c61aea3-3cba-35f3-8971-df42cd5b9b1a
  7c30c3fc-ea17-38d8-9c52-c75ccb112253
  a2f568b5-060f-33f0-9175-7e2062d86b6c
  b9f73e2a-292a-3876-b363-3ebb94584c7a
  cea5f5c2-e786-30f5-8305-baead8923063
  6b0cc3b0-2802-33a7-b885-f1f1409345ac
  7cb4b11f-3872-3825-83b5-622e1a2cdb28
  a359e053-a350-36cf-ab1d-a7980afaffa2
  c654b457-11d4-393c-a638-188855c8f2e5
  f41d0e8f-856e-3f7d-a3f9-ff5ba7c8e06d
  6f2f7d1e-8ded-35c5-ba83-3ca906b05127
  8aad8778-73ce-3fa0-93c7-804ac998667d
  b51561d9-08b0-3599-bc78-016f1441bb91
  c990cafc-f96c-3107-b213-01d217b11272
)

export DATASET_NAME="sensor"  # sensor, lidar, motion_forecasting or tbv.
export TARGET_DIR="$HOME/Downloads/Datasets/av2/train"

for SEQUENCE_ID in $SEQUENCE_IDS; do
    echo "Downloading sequence argoverse/datasets/av2/$DATASET_NAME/train/$SEQUENCE_ID/* to $TARGET_DIR/$SEQUENCE_ID"
    s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/train/$SEQUENCE_ID/*" "$TARGET_DIR/$SEQUENCE_ID"
done
```

### Downtown Split

```bash
#!/usr/bin/env bash

# Dataset URIs
# s3://argoverse/datasets/av2/sensor/

SEQUENCE_IDS=(

)

export DATASET_NAME="sensor"  # sensor, lidar, motion_forecasting or tbv.
export TARGET_DIR="$HOME/Downloads/Datasets/av2/train"

for SEQUENCE_ID in $SEQUENCE_IDS; do
    echo "Downloading sequence argoverse/datasets/av2/$DATASET_NAME/train/$SEQUENCE_ID/* to $TARGET_DIR/$SEQUENCE_ID"
    s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/train/$SEQUENCE_ID/*" "$TARGET_DIR/$SEQUENCE_ID"
done
```