from collections.abc import Collection, Sequence
from typing import Iterable, Mapping

import numpy as np
import torch
from neuralogic.core.builder.builder import NeuralSample

from lib.nn.definitions.ops import AggregationDef, TransformationDef
from lib.nn.sources.base import LayerDefinition, WeightDefinition
from lib.nn.sources.base_impl import BaseWeightDefinition
from lib.nn.sources.minimal_api.base import MinimalAPINetwork
from lib.nn.sources.minimal_api.internal.java import (
    JavaNeuron,
    JavaWeight,
    compute_java_neurons_per_layer,
    discover_layers,
    get_aggregation,
    get_transformation,
)
from lib.nn.topological.settings import Settings
from lib.utils import LambdaIterable, MapCollection, MapSequence, cache, value_to_numpy, value_to_tensor


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
        return value_to_numpy(self._java_weight.value)

    def get_value_torch(self) -> torch.Tensor:
        return value_to_tensor(self._java_weight.value)

    def __hash__(self) -> int:
        return hash((self.learnable, self.id))

    def __eq__(self, value: object) -> bool:
        return isinstance(value, WeightDefinition) and self.learnable == value.learnable and self.id == value.id


class MinimalAPIJavaNetwork(MinimalAPINetwork[Sequence[JavaNeuron]]):
    """Minimal API for a neural network representation from NeuraLogic Java library.

    See documentation for `MinimalAPINetwork` for details.
    """

    def __init__(self, samples: Sequence[NeuralSample | JavaNeuron], settings: Settings) -> None:
        self._samples = samples
        self._settings = settings
        self._java_neurons_per_layer = compute_java_neurons_per_layer(samples)

    @cache
    def get_layers(self) -> Sequence[LayerDefinition]:
        return discover_layers(
            self._samples,
            check_same_layers_assumption=self._settings.check_same_layers_assumption,
        )

    @cache
    def get_layers_map(self) -> Mapping[int, LayerDefinition]:
        return {l.id: l for l in self.get_layers()}

    def get_layer_neurons(self, layer_id: int) -> Sequence[JavaNeuron]:
        return self._java_neurons_per_layer[layer_id]

    def get_ids(self, neurons: Sequence[JavaNeuron]) -> Sequence[int]:
        return MapSequence(lambda n: int(n.getIndex()), neurons)

    def get_inputs(self, neurons: Sequence[JavaNeuron]) -> Sequence[JavaNeuron]:
        return [inp for n in neurons for inp in n.getInputs()]

    def get_input_lengths(self, neurons: Sequence[JavaNeuron]) -> Sequence[int]:
        return MapSequence(lambda n: int(len(n.getInputs())), neurons)

    def get_input_weights(self, neurons: Sequence[JavaNeuron]) -> Iterable[WeightDefinition]:
        return LambdaIterable(lambda: (_JavaWeightDefinition(w) for n in neurons for w in n.getWeights()))

    def get_biases(self, neurons: Sequence[JavaNeuron]) -> Sequence[WeightDefinition]:
        return MapSequence(lambda n: _JavaWeightDefinition(n.getOffset()), neurons)

    def get_values_numpy(self, neurons: Sequence[JavaNeuron]) -> Collection[np.ndarray]:
        return MapCollection(lambda n: value_to_numpy(n.getRawState().getValue()), neurons)

    def get_values_torch(self, neurons: Sequence[JavaNeuron]) -> Collection[torch.Tensor]:
        return MapCollection(lambda n: value_to_tensor(n.getRawState().getValue()), neurons)

    def get_transformations(self, neurons: Sequence[JavaNeuron]) -> Sequence[TransformationDef | None]:
        return MapSequence(get_transformation, neurons)

    def get_aggregations(self, neurons: Sequence[JavaNeuron]) -> Sequence[AggregationDef | None]:
        return MapSequence(get_aggregation, neurons)

    def slice(self, neurons: Sequence[JavaNeuron], sl: slice) -> Sequence[JavaNeuron]:
        return neurons[sl]

    def select_ids(self, neurons: Sequence[JavaNeuron], ids: Sequence[int]) -> Sequence[JavaNeuron]:
        ids_all = self.get_ids(neurons)
        map = dict(zip(ids_all, neurons))
        return [map[id] for id in ids]
