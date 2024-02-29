from typing import Collection, Iterable, Iterator, Sequence

import numpy as np
import torch
from neuralogic.core.builder.builder import NeuralSample

from lib.nn.sources.base_source import (
    BaseLayerNeurons,
    BaseNeuralNetworkDefinition,
    BaseNeurons,
    BaseWeightDefinition,
)
from lib.nn.sources.source import (
    LayerDefinition,
    LayerDefinitions,
    LayerNeurons,
    LayerOrdinal,
    Neurons,
    Ordinals,
    WeightDefinition,
    get_layer_id,
)
from lib.nn.sources.utils import LayerDefinitionsImpl, OrdinalsDict, OrdinalsProxy
from lib.nn.topological.settings import Settings
from lib.utils import LambdaIterable, MapCollection, cache, value_to_numpy, value_to_tensor

from .internal.java_source import (
    JavaNeuron,
    JavaWeight,
    compute_java_neurons,
    compute_java_neurons_per_layer,
    compute_java_ordinals_for_layer,
    discover_layers,
)


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


class _JavaNeurons(BaseNeurons):
    def __init__(
        self,
        network: "JavaNeuralNetworkDefinition",
        java_neurons: Collection[JavaNeuron],
        ordinals: Ordinals,
    ) -> None:
        self._network = network
        self._java_neurons = java_neurons
        self._ordinals = ordinals
        self._input_lengths = MapCollection(lambda n: int(len(n.getInputs())), self._java_neurons)

    @property
    def ordinals(self) -> Ordinals:
        return self._ordinals

    @property
    def ids(self) -> Collection[int]:
        return self._ordinals.ids()

    @property
    def input_lengths(self) -> Collection[int]:
        return self._input_lengths

    @property
    @cache
    def inputs(self) -> "Neurons":
        input_java_neurons = [inp for n in self._java_neurons for inp in n.getInputs()]
        input_ids = MapCollection(lambda n: int(n.getIndex()), input_java_neurons)
        input_ordinals = OrdinalsProxy(
            network_ordinals=self._network.ordinals,
            ids=input_ids,
            items=MapCollection(lambda id: (id, self._network.ordinals[id]), input_ids),
            values=MapCollection(lambda id: self._network.ordinals[id], input_ids),
        )
        return _JavaNeurons(self._network, input_java_neurons, input_ordinals)

    def __len__(self) -> int:
        return len(self._ordinals)

    @property
    def input_weights(self) -> Iterable[WeightDefinition]:
        return LambdaIterable(lambda: (_JavaWeightDefinition(w) for n in self._java_neurons for w in n.getWeights()))

    def get_values_numpy(self) -> Collection[np.ndarray]:
        return MapCollection(lambda n: value_to_numpy(n.getRawState().getValue()), self._java_neurons)

    def get_values_torch(self) -> Collection[torch.Tensor]:
        return MapCollection(lambda n: value_to_tensor(n.getRawState().getValue()), self._java_neurons)


class _JavaLayerNeurons(_JavaNeurons, BaseLayerNeurons):
    def __init__(
        self,
        network: "JavaNeuralNetworkDefinition",
        java_neurons: Collection[JavaNeuron],
        ordinals: Ordinals,
        layer: LayerDefinition,
    ) -> None:
        super().__init__(network, java_neurons, ordinals)
        self._layer = layer

    @property
    def layer(self) -> LayerDefinition:
        return self._layer


class JavaNeuralNetworkDefinition(BaseNeuralNetworkDefinition):
    def __init__(self, samples: Sequence[NeuralSample | JavaNeuron], settings: Settings):
        self._layer_definitions = LayerDefinitionsImpl.from_iter(
            discover_layers(samples, settings.check_same_layers_assumption)
        )
        self._java_neurons_per_layer = compute_java_neurons_per_layer(samples)
        self._java_neurons = compute_java_neurons(self._java_neurons_per_layer, self._layer_definitions)
        self._ordinals_per_layer = {
            ld.id: OrdinalsDict(compute_java_ordinals_for_layer(ld, self._java_neurons_per_layer[ld.id], settings))
            for ld in self._layer_definitions
        }
        self._ordinals = OrdinalsDict(
            {id: o for layer_ordinals in self._ordinals_per_layer.values() for id, o in layer_ordinals.items()}
        )
        self._items = MapCollection(lambda ld: (ld, self[ld]), self.layers)

    @property
    def ordinals(self) -> Ordinals:
        return self._ordinals

    @cache
    def __getitem__(self, layer: int | LayerDefinition) -> LayerNeurons:
        layer_id = get_layer_id(layer)

        java_neurons = self._java_neurons_per_layer[layer_id]
        ordinals = self._ordinals_per_layer[layer_id]
        layer = self.layers[layer_id]

        return _JavaLayerNeurons(self, java_neurons, ordinals, layer)

    def __len__(self) -> int:
        return len(self._layer_definitions)

    @property
    def layers(self) -> LayerDefinitions:
        return self._layer_definitions

    def items(self) -> Collection[tuple[LayerDefinition, LayerNeurons]]:
        return self._items

    def __iter__(self) -> Iterator[LayerNeurons]:
        yield from (self[ld] for ld in self.layers)

    def __contains__(self, o: object) -> bool:
        if isinstance(o, (LayerDefinition, int)):
            return o in self.layers

        if isinstance(o, LayerOrdinal):
            return o in self._ordinals

        raise ValueError()
