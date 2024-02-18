import numpy as np
import torch

DTYPE_TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def value_to_numpy(java_value, dtype: torch.dtype | None = None) -> np.ndarray:
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype not in DTYPE_TORCH_TO_NUMPY:
        raise NotImplementedError(f"Conversion from {dtype} to numpy equivalent not yet implemented.")

    np_dtype = DTYPE_TORCH_TO_NUMPY[dtype]

    arr = np.asarray(java_value.getAsArray(), dtype=np_dtype)
    arr = arr.reshape(java_value.size())
    return arr


def value_to_tensor(java_value, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.tensor(value_to_numpy(java_value, dtype))


def atleast_3d_rev(tensor: torch.Tensor) -> torch.Tensor:
    dim = tensor.dim()

    if dim == 0:
        return tensor.reshape([1, 1, 1])
    elif dim == 1:
        return tensor.reshape([-1, 1, 1])
    elif dim == 2:
        return tensor.unsqueeze(-1)
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


