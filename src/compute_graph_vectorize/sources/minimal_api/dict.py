from dataclasses import asdict, dataclass, field
from typing import Collection, Iterable, Mapping, Sequence

import numpy as np
import torch

from compute_graph_vectorize.model.ops import AggregationDef, TransformationDef
from compute_graph_vectorize.sources.base import LayerDefinition, WeightDefinition
from compute_graph_vectorize.sources.base_impl import BaseWeightDefinition
from compute_graph_vectorize.sources.minimal_api.base import MinimalAPINetwork
from compute_graph_vectorize.utils import LambdaIterable, MapCollection, MapSequence, cache


class _Value:
    def __init__(self, value: np.ndarray | torch.Tensor) -> None:
        if isinstance(value, np.ndarray):
            self._value_np = value
            self._value_torch = None
        elif isinstance(value, torch.Tensor):
            self._value_np = None
            self._value_torch = value

    @cache
    def get_value_numpy(self) -> np.ndarray:
        if self._value_np is None:
            assert self._value_torch is not None
            self._value_np = self._value_torch.numpy()

        return self._value_np

    @cache
    def get_value_torch(self) -> torch.Tensor:
        if self._value_torch is None:
            assert self._value_np is not None
            if isinstance(self._value_np.dtype, np.floating):
                self._value_torch = torch.tensor(self._value_np, dtype=torch.get_default_dtype())
            else:
                self._value_torch = torch.tensor(self._value_np)

        return self._value_torch


class WeightDefinitionImpl(BaseWeightDefinition):
    __slots__ = ("_learnable", "_id", "_value_np", "_value_torch")

    def __init__(self, id: int, value: np.ndarray | torch.Tensor, learnable: bool) -> None:
        self._id = id
        self._learnable = learnable
        self._value = _Value(value)

    @property
    def learnable(self) -> bool:
        return self._learnable

    @property
    def id(self) -> int:
        return self._id

    def get_value_numpy(self) -> np.ndarray:
        return self._value.get_value_numpy()

    def get_value_torch(self) -> torch.Tensor:
        return self._value.get_value_torch()

    def __hash__(self) -> int:
        return hash((self.learnable, self.id))

    def __eq__(self, value: object) -> bool:
        return isinstance(value, WeightDefinition) and self.id == value.id and self.learnable == value.learnable


_ZERO_WEIGHT = WeightDefinitionImpl(id=-2, value=np.zeros(0), learnable=False)


@dataclass(frozen=True)
class Neuron:
    id: int
    transformation: TransformationDef | None
    aggregation: AggregationDef | None
    inputs: Sequence[int] = field(default_factory=lambda: [], hash=False)
    weights: Sequence[WeightDefinition] = field(default_factory=lambda: [], hash=False)
    bias: WeightDefinition = field(default_factory=lambda: _ZERO_WEIGHT, hash=False)
    value: np.ndarray = field(default_factory=lambda: np.zeros(0), hash=False)

    @classmethod
    def create_from(cls, other: "Neuron", **kwargs):
        the_kwargs = asdict(other)
        the_kwargs.update(kwargs)
        return cls(**the_kwargs)


class MinimalAPIDictNetwork(MinimalAPINetwork[Sequence[Neuron]]):
    """Minimal API for a neural network representation kept directly in Python.

    See documentation for `MinimalAPINetwork` for details.
    """

    def __init__(
        self,
        layers: Sequence[LayerDefinition],
        neurons: Sequence[Sequence[Neuron]] | Mapping[str, Sequence[Neuron]],
    ) -> None:
        self._layers = layers
        self._layers_map = {l.id: l for l in self.get_layers()}

        self._neurons_per_layer: Mapping[str, Sequence[Neuron]]
        if isinstance(neurons, Sequence):
            self._neurons_per_layer = {ld.id: ns for ld, ns in zip(self._layers, neurons)}
        else:
            self._neurons_per_layer = neurons

        self._neurons: Mapping[int, Neuron] = {n.id: n for ns in self._neurons_per_layer.values() for n in ns}

    def get_layers(self) -> Sequence[LayerDefinition]:
        return self._layers

    def get_layers_map(self) -> Mapping[str, LayerDefinition]:
        return self._layers_map

    def get_layer_neurons(self, layer_id: str) -> Sequence[Neuron]:
        return self._neurons_per_layer[layer_id]

    def get_ids(self, neurons: Sequence[Neuron]) -> Sequence[int]:
        return MapSequence(lambda n: n.id, neurons)

    def get_inputs(self, neurons: Sequence[Neuron]) -> Sequence[Neuron]:
        return [self._neurons[inp] for n in neurons for inp in n.inputs]

    def get_input_lengths(self, neurons: Sequence[Neuron]) -> Sequence[int]:
        return MapSequence(lambda n: len(n.inputs), neurons)

    def get_input_weights(self, neurons: Sequence[Neuron]) -> Iterable[WeightDefinition]:
        return LambdaIterable(lambda: (w for n in neurons for w in n.weights))

    def get_biases(self, neurons: Sequence[Neuron]) -> Sequence[WeightDefinition]:
        return MapSequence(lambda n: n.bias, neurons)

    def get_values_numpy(self, neurons: Sequence[Neuron]) -> Collection[np.ndarray]:
        return MapCollection(lambda n: n.value, neurons)

    def get_values_torch(self, neurons: Sequence[Neuron]) -> Collection[torch.Tensor]:
        return MapCollection(lambda n: torch.tensor(n.value, dtype=torch.get_default_dtype()), neurons)

    def get_transformations(self, neurons: Sequence[Neuron]) -> Sequence[TransformationDef | None]:
        return MapSequence(lambda n: n.transformation, neurons)

    def get_aggregations(self, neurons: Sequence[Neuron]) -> Sequence[AggregationDef | None]:
        return MapSequence(lambda n: n.aggregation, neurons)

    def slice(self, neurons: Sequence[Neuron], sl: slice) -> Sequence[Neuron]:
        return neurons[sl]

    def select_ids(self, neurons: Sequence[Neuron], ids: Sequence[int]) -> Sequence[Neuron]:
        ids_all = self.get_ids(neurons)
        map = dict(zip(ids_all, neurons))
        return [map[id] for id in ids]
