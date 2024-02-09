from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LayerPipe(Protocol):
    @property
    def delegate(self) -> torch.nn.Module:
        ...


class LayerInputPipe(torch.nn.Module):
    def __init__(self, layer: int, delegate: torch.nn.Module) -> None:
        super().__init__()
        self.layer_index = layer
        self.delegate = delegate

    def forward(self, layer_values: dict[int, torch.Tensor]):
        return self.delegate(layer_values[self.layer_index])

    def extra_repr(self) -> str:
        return f"layer={self.layer_index},"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}[{self.layer_index}]({repr(self.delegate)})"


class LayerOutputPipe(torch.nn.Module):
    def __init__(self, layer: int, delegate: torch.nn.Module) -> None:
        super().__init__()
        self.delegate = delegate
        self.layer_index = layer

    def forward(self, layer_values: dict[int, torch.Tensor] | None = None):
        # TODO: autodetect in preprocessing which layers can be thrown away when for saving memory
        if layer_values is None:
            layer_values = {}

        layer_values[self.layer_index] = self.delegate(layer_values)
        return layer_values

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}[{self.layer_index}]({repr(self.delegate)})"
