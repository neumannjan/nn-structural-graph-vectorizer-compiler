from collections.abc import Collection
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, NamedTuple, OrderedDict, Protocol, runtime_checkable

import numpy as np
import torch

LayerType = Literal["FactLayer", "WeightedAtomLayer", "WeightedRuleLayer", "AggregationLayer"]


@dataclass(frozen=True)
class LayerDefinition:
    id: int
    type: LayerType


def get_layer_id(layer: int | LayerDefinition):
    if isinstance(layer, LayerDefinition):
        layer_id = layer.id
    else:
        layer_id = layer
    return layer_id


class LayerOrdinal(NamedTuple):
    layer: int
    ordinal: int

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer={self.layer}, ordinal={self.ordinal})"


class Ordinals(Protocol):
    def ids(self) -> Collection[int]:
        ...

    def items(self) -> Collection[tuple[int, LayerOrdinal]]:
        ...

    def values(self) -> Collection[LayerOrdinal]:
        ...

    def __iter__(self) -> Iterator[LayerOrdinal]:
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, o: object) -> bool:
        ...

    def __getitem__(self, id: int) -> LayerOrdinal:
        ...


class LayerDefinitions(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, layer_id: int) -> LayerDefinition:
        ...

    def __iter__(self) -> Iterator[LayerDefinition]:
        ...

    def __contains__(self, ld: object) -> bool:
        ...

    def as_list(self) -> list[LayerDefinition]:
        ...

    def as_dict(self) -> OrderedDict[int, LayerDefinition]:
        ...


@runtime_checkable
class WeightDefinition(Protocol):
    @property
    def learnable(self) -> bool:
        ...

    @property
    def id(self) -> int:
        ...

    def get_value_numpy(self) -> np.ndarray:
        ...

    def get_value_torch(self) -> torch.Tensor:
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, value: object) -> bool:
        ...


class Neurons(Protocol):
    @property
    def ordinals(self) -> Ordinals:
        ...

    @property
    def ids(self) -> Collection[int]:
        ...

    @property
    def input_lengths(self) -> Collection[int]:
        ...

    @property
    def inputs(self) -> "Neurons":
        ...

    def __len__(self) -> int:
        ...

    @property
    def input_weights(self) -> Iterable[WeightDefinition]:
        ...

    def get_values_numpy(self) -> Collection[np.ndarray]:
        ...

    def get_values_torch(self) -> Collection[torch.Tensor]:
        ...


class LayerNeurons(Neurons, Protocol):
    @property
    def layer(self) -> LayerDefinition:
        ...


class NeuralNetworkDefinition(Protocol):
    @property
    def ordinals(self) -> Ordinals:
        ...

    def __getitem__(self, layer: int | LayerDefinition) -> LayerNeurons:
        ...

    def __len__(self) -> int:
        ...

    @property
    def layers(self) -> LayerDefinitions:
        ...

    def items(self) -> Collection[tuple[LayerDefinition, LayerNeurons]]:
        ...

    def __iter__(self) -> Iterator[LayerNeurons]:
        ...

    def __contains__(self, o: object) -> bool:
        ...
