# [KITTI](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)

Download the dataset to a location of your convenience. You can later adjust the data path in the preprocessing script.

## Preprocessing

By default we assume the following dataset location.
```
data/
    KITTI/
        tracking/
            training/
                image_02/
                ...
```

You can then process the data with the following commands:

```
mp-process kitti --sequence 0001
mp-process kitti --sequence 0002
mp-process kitti --sequence 0006
```