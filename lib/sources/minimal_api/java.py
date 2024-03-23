from collections.abc import Collection, Sequence
from typing import Iterable, Mapping

import numpy as np
import torch
from neuralogic.core.builder.builder import NeuralSample

from lib.nn.definitions.ops import AggregationDef, TransformationDef
from lib.nn.definitions.settings import Settings
from lib.sources.base import LayerDefinition, LayerType, WeightDefinition, is_weighted
from lib.sources.base_impl import BaseWeightDefinition
from lib.sources.minimal_api.base import MinimalAPINetwork
from lib.sources.minimal_api.internal.java import (
    JavaNeuron,
    JavaWeight,
    compute_java_neurons_per_layer,
    get_aggregation,
    get_transformation,
    java_value_to_numpy,
    java_value_to_tensor,
)
from lib.utils import LambdaIterable, MapCollection, MapSequence, cache


class _JavaWeightDefinition(BaseWeightDefinition):
    __slots__ = ["_java_weight"]

    def __init__(self, java_weight: JavaWeight) -> None:
        self._java_weight = java_weight

    @property
    def learnable(self) -> bool:
        return bool(self._java_weight.isLearnable())

    @property
    def id(self) -> int:
        return int(self._java_weight.index)

    def get_value_numpy(self) -> np.ndarray:
        return java_value_to_numpy(self._java_weight.value)

    def get_value_torch(self) -> torch.Tensor:
        return java_value_to_tensor(self._java_weight.value)

    def __hash__(self) -> int:
        return hash((self.learnable, self.id))

    def __eq__(self, value: object) -> bool:
        return isinstance(value, WeightDefinition) and self.learnable == value.learnable and self.id == value.id


class _JavaNeuronsPointer:
    __slots__ = ("_layer_type", "_n")

    def __init__(self, layer_type: LayerType | None, neurons: Sequence[JavaNeuron]) -> None:
        self._layer_type: LayerType | None = layer_type
        self._n = neurons


class MinimalAPIJavaNetwork(MinimalAPINetwork[_JavaNeuronsPointer]):
    """Minimal API for a neural network representation from NeuraLogic Java library.

    See documentation for `MinimalAPINetwork` for details.
    """

    def __init__(self, samples: Sequence[NeuralSample | JavaNeuron], settings: Settings) -> None:
        self._samples = samples
        self._settings = settings
        self._java_neurons_per_layer, self._layers = compute_java_neurons_per_layer(samples)

    def get_layers(self) -> Sequence[LayerDefinition]:
        return self._layers

    @cache
    def get_layers_map(self) -> Mapping[int, LayerDefinition]:
        return {l.id: l for l in self.get_layers()}

    def get_layer_neurons(self, layer_id: int) -> _JavaNeuronsPointer:
        return _JavaNeuronsPointer(self.get_layers_map()[layer_id].type, self._java_neurons_per_layer[layer_id])

    def get_ids(self, neurons: _JavaNeuronsPointer) -> Sequence[int]:
        return MapSequence(lambda n: int(n.getIndex()), neurons._n)

    def get_inputs(self, neurons: _JavaNeuronsPointer) -> _JavaNeuronsPointer:
        return _JavaNeuronsPointer(None, [inp for n in neurons._n for inp in n.getInputs()])

    def get_input_lengths(self, neurons: _JavaNeuronsPointer) -> Sequence[int]:
        return MapSequence(lambda n: int(len(n.getInputs())), neurons._n)

    def get_input_weights(self, neurons: _JavaNeuronsPointer) -> Iterable[WeightDefinition]:
        if neurons._layer_type is not None and is_weighted(neurons._layer_type):
            return LambdaIterable(lambda: (_JavaWeightDefinition(w) for n in neurons._n for w in n.getWeights()))
        return []

    def get_biases(self, neurons: _JavaNeuronsPointer) -> Sequence[WeightDefinition]:
        return MapSequence(lambda n: _JavaWeightDefinition(n.getOffset()), neurons._n)

    def get_values_numpy(self, neurons: _JavaNeuronsPointer) -> Collection[np.ndarray]:
        return MapCollection(lambda n: java_value_to_numpy(n.getRawState().getValue()), neurons._n)

    def get_values_torch(self, neurons: _JavaNeuronsPointer) -> Collection[torch.Tensor]:
        return MapCollection(lambda n: java_value_to_tensor(n.getRawState().getValue()), neurons._n)

    def get_transformations(self, neurons: _JavaNeuronsPointer) -> Sequence[TransformationDef | None]:
        return MapSequence(get_transformation, neurons._n)

    def get_aggregations(self, neurons: _JavaNeuronsPointer) -> Sequence[AggregationDef | None]:
        return MapSequence(get_aggregation, neurons._n)

    def slice(self, neurons: _JavaNeuronsPointer, sl: slice) -> _JavaNeuronsPointer:
        return _JavaNeuronsPointer(neurons._layer_type, neurons._n[sl])

    def select_ids(self, neurons: _JavaNeuronsPointer, ids: Sequence[int]) -> _JavaNeuronsPointer:
        ids_all = self.get_ids(neurons)
        map = dict(zip(ids_all, neurons._n))
        return _JavaNeuronsPointer(neurons._layer_type, [map[id] for id in ids])
