import numpy as np
import torch


def value_to_numpy(java_value) -> np.ndarray:
    arr = np.asarray(java_value.getAsArray())
    arr = arr.reshape(java_value.size())
    return arr


def value_to_tensor(java_value) -> torch.Tensor:
    return torch.tensor(value_to_numpy(java_value))


def atleast_3d_rev(tensor: torch.Tensor) -> torch.Tensor:
    dim = tensor.dim()

    if dim == 0:
        return tensor.reshape([1, 1, 1])
    elif dim == 1:
        return tensor.reshape([-1, 1, 1])
    elif dim == 2:
        return tensor.reshape([*tensor.shape, 1])
    else:
        return tensor


def atleast_2d_rev(tensor: torch.Tensor) -> torch.Tensor:
    dim = tensor.dim()

    if dim == 0:
        return tensor.reshape([1, 1])
    elif dim == 1:
        return tensor.reshape([-1, 1])
    else:
        return tensor


def expand_diag(tensor: torch.Tensor, n: int) -> torch.Tensor:
    tensor = torch.squeeze(tensor)
    dim = tensor.dim()
    if dim > 2:
        raise ValueError()

    if dim == 2:
        return tensor

    if dim == 0:
        tensor = torch.atleast_1d(tensor).expand([n])

    return torch.diag(tensor)


