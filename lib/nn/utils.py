from typing import Iterable, Protocol, overload, runtime_checkable

import torch
from torch.jit import unused


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


@runtime_checkable
class ShapeTransformable(Protocol):
    @overload
    def compute_output_shape(self) -> list[int]: ...
    @overload
    def compute_output_shape(self, shape: list[int]) -> list[int]: ...
    @overload
    def compute_output_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]: ...


class Identity(torch.nn.Module, ShapeTransformable):
    def forward(self, x):
        return x

    @unused
    def compute_output_shape(self, shape: list[int]) -> list[int]:
        return shape


class Sequential(torch.nn.Module, ShapeTransformable):
    def __init__(self, modules: Iterable[ShapeTransformable]) -> None:
        super().__init__()
        self.the_modules = torch.nn.ModuleList(modules)

    def forward(self, input):
        for module in self.the_modules:
            input = module(input)
        return input

    @unused
    def compute_output_shape(self, shape_like) -> list[int]:
        for module in self.the_modules:
            shape_like = module.compute_output_shape(shape_like)

        return shape_like

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({repr(self.the_modules)})"


class SingleLayerOperation(torch.nn.Module, ShapeTransformable):
    def __init__(self, input_layer: int, the_module: ShapeTransformable) -> None:
        super().__init__()
        self.input_layer = input_layer
        self.delegate = the_module

    @unused
    def compute_output_shape(self, shapes_or_shape: dict[str, list[int]] | list[int]) -> list[int]:
        if isinstance(shapes_or_shape, dict):
            shape = shapes_or_shape[str(self.input_layer)]
        else:
            shape = shapes_or_shape

        return self.delegate.compute_output_shape(shape)

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        x = layer_values[str(self.input_layer)]
        x = self.delegate(x)
        return x

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}[{self.input_layer}]({repr(self.delegate)})"
