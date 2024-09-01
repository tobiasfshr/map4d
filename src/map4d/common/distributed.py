"""Custom Utils for DDP."""
import pickle
from typing import Any

import torch
import torch.distributed as dist
from nerfstudio.utils.rich_utils import CONSOLE


def dict_to_cpu(dict_to_convert: dict):
    """Put all values inside a possibly nested dict on cpu."""
    # put all values in metrics dict to cpu if they are tensor
    for key, val in dict_to_convert.items():
        if isinstance(val, torch.Tensor):
            dict_to_convert[key] = val.cpu().item()
        elif isinstance(val, dict):
            dict_to_cpu(val)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, torch.Tensor):
                    val[i] = item.cpu().item()
                elif isinstance(item, dict):
                    dict_to_cpu(item)


def serialize_to_tensor(data: Any) -> torch.Tensor:
    """Serialize arbitrary picklable data to a torch.Tensor.

    Args:
        data (Any): The data to serialize.

    Returns:
        torch.Tensor: The serialized data as a torch.Tensor.
    """
    backend = dist.get_backend()
    assert backend in {
        "gloo",
        "nccl",
    }, "_serialize_to_tensor only supports gloo and nccl backends."
    device = torch.device("cpu") if backend == "gloo" else torch.cuda.current_device()

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        CONSOLE.log(
            "WARNING: You're trying to all-gather %.2f GB of data on device %s",
            len(buffer) / (1024**3),
            device,
        )
    storage = torch.UntypedStorage.from_buffer(buffer, dtype=torch.uint8)
    tensor = torch.tensor(storage, dtype=torch.uint8, device=device)
    return tensor


def pad_to_largest_tensor(tensor: torch.Tensor, world_size: int) -> tuple[list[int], torch.Tensor]:
    """Pad tensor to largest size among the tensors in each process.

    Args:
        tensor: tensor to be padded.
        world_size: number of processes.

    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    local_size_list = [local_size.clone() for _ in range(world_size)]
    dist.all_gather(local_size_list, local_size)
    size_list = [int(size.item()) for size in local_size_list]
    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_object_list(data: list[Any], world_size: int) -> list[Any]:
    """Run all_gather on arbitrary picklable data.

    Args:
        data: data to be gathered
        world_size: number of processes

    Returns:
        list[Any]: full list of data gathered from each process
    """
    if world_size == 1:
        return data

    # encode
    tensor = serialize_to_tensor(data)
    size_list, tensor = pad_to_largest_tensor(tensor, world_size)
    tensor_list = [tensor.clone() for _ in range(world_size)]

    # gather (world_size, N)
    dist.all_gather(tensor_list, tensor)

    # decode
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    # fuse lists into a single list
    data_list = [item for sublist in data_list for item in sublist]
    return data_list
