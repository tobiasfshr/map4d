

# [VKITTI2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)

## Preprocessing

Download the dataset to a location of your convenience. You can later adjust the data path in the preprocessing script.

By default we assume the following dataset location.
```
data/
    VKITTI2/
        Scene01/
        ...
```

You can then process the data with the following commands:

```
mp-process vkitti2 --sequence 02
mp-process vkitti2 --sequence 06
mp-process vkitti2 --sequence 18
```