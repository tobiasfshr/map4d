# [Waymo](https://waymo.com/open/)

## Requirements
Please the Waymo API install via

```
pip install waymo-open-dataset-tf-2-11-0==1.6.1 --no-deps
```

Note that due to a dependency conflict with numpy, you need to install e.g. tensorflow manually after.

## Download & Preprocessing

We provide data download and preprocessing of the full Dynamic-32 split from [EmerNeRF](https://emernerf.github.io/) via a single command:

```
mp-process waymo
```

By default, this will download and the data to the following location:

```
data/
    waymo/
        raw/
            segment-...
        processed/
            ...
        metadata_segment-...
```

You can adjust the path with the `--data` option.