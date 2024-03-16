import torch

@torch.jit.script
def _broadcast_shapes_compiled(shapes: list[list[int]]) -> list[int]:
    max_len = 0
    for shape in shapes:
        s = len(shape)
        if max_len < s:
            max_len = s

    result = [1] * max_len
    for shape in shapes:
        for i in range(-1, -1 - len(shape), -1):
            if shape[i] < 0:
                raise RuntimeError(
                    "Trying to create tensor with negative dimension ({}): ({})".format(shape[i], shape[i])
                )
            if shape[i] == 1 or shape[i] == result[i]:
                continue
            if result[i] != 1:
                raise RuntimeError("Shape mismatch: objects cannot be broadcast to a single shape")
            result[i] = shape[i]

    return list(result)

def broadcast_shapes_compiled(shapes: list[list[int]]) -> list[int]:
    if torch.jit.is_scripting():
        return _broadcast_shapes_compiled(shapes)
    else:
        return torch.broadcast_shapes(*shapes)
