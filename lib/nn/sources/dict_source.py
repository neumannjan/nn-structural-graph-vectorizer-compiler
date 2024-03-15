from dataclasses import asdict, dataclass, field
from typing import Collection, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from lib.nn.definitions.ops import TransformationDef
from lib.nn.sources.base_source import BaseNeuralNetworkDefinition, BaseNeurons, BaseWeightDefinition
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
from lib.utils import MapCollection, cache


class WeightDefinitionImpl(BaseWeightDefinition):
    __slots__ = ("_learnable", "_id", "_value_np", "_value_torch")

    def __init__(self, id: int, value: np.ndarray | torch.Tensor, learnable: bool) -> None:
        self._id = id
        if isinstance(value, np.ndarray):
            self._value_np = value
            self._value_torch = None
        elif isinstance(value, torch.Tensor):
            self._value_np = None
            self._value_torch = value
        self._learnable = learnable

    @property
    def learnable(self) -> bool:
        return self._learnable

    @property
    def id(self) -> int:
        return self._id

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

    def __hash__(self) -> int:
        return hash((self.learnable, self.id))

    def __eq__(self, value: object) -> bool:
        return isinstance(value, WeightDefinition) and self.id == value.id and self.learnable == value.learnable


_ZERO_WEIGHT = WeightDefinitionImpl(id=-2, value=np.zeros(0), learnable=False)


@dataclass(frozen=True)
class Neuron:
    id: int
    transformation: TransformationDef | None
    inputs: Sequence[int] = field(default_factory=lambda: [], hash=False)
    weights: Sequence[WeightDefinition] = field(default_factory=lambda: [], hash=False)
    bias: WeightDefinition = field(default_factory=lambda: _ZERO_WEIGHT, hash=False)
    value: np.ndarray = field(default_factory=lambda: np.zeros(0), hash=False)

    @classmethod
    def create_from(cls, other: "Neuron", **kwargs):
        the_kwargs = asdict(other)
        the_kwargs.update(kwargs)
        return cls(**the_kwargs)


class _NeuronsList(BaseNeurons):
    def __init__(
        self,
        network: "NeuralNetworkDefinitionDict",
        ordinals: Ordinals,
        neurons: Collection[Neuron],
    ) -> None:
        self._network = network
        self._ordinals = ordinals
        self._neurons = neurons
        self._ids = MapCollection(lambda n: n.id, neurons)
        self._input_lengths = MapCollection(lambda n: len(n.inputs), neurons)
        self._input_weights = MapCollection(lambda n: n.weights, neurons)

    @property
    def ordinals(self) -> Ordinals:
        return self._ordinals

    @property
    @cache
    def _inputs(self):
        return [inp for n in self._neurons for inp in n.inputs]

    @property
    def ids(self) -> Collection[int]:
        return self._ids

    @property
    def input_lengths(self) -> Collection[int]:
        return self._input_lengths

    @property
    def inputs(self) -> "Neurons":
        return _NeuronsList(
            self._network,
            ordinals=OrdinalsProxy(
                self._network.ordinals,
                ids=MapCollection(lambda n_id: n_id, self._inputs),
                items=MapCollection(lambda n_id: (n_id, self._network.ordinals[n_id]), self._inputs),
                values=MapCollection(lambda n_id: self._network.ordinals[n_id], self._inputs),
            ),
            neurons=MapCollection(lambda n_id: self._network._neurons[n_id], self._inputs),
        )

    def __len__(self) -> int:
        return len(self._neurons)

    def biases(self) -> Collection[WeightDefinition]:
        return MapCollection(lambda n: n.bias, self._neurons)

    @property
    def input_weights(self) -> Iterable[WeightDefinition]:
        yield from (w for n in self._neurons for w in n.weights)

    def get_values_numpy(self) -> Collection[np.ndarray]:
        return MapCollection(lambda n: n.value, self._neurons)

    def get_values_torch(self) -> Collection[torch.Tensor]:
        return MapCollection(lambda n: torch.tensor(n.value), self._neurons)

    def get_transformations(self) -> Collection[TransformationDef | None]:
        return MapCollection(lambda n: n.transformation, self._neurons)

    def gather(self, ids: Sequence[int]) -> "Neurons":
        id_set = set(ids)
        ordinals_neurons = [(o, n) for o, n in zip(self._ordinals, self._neurons) if n.id in id_set]

        return _NeuronsList(
            network=self._network,
            ordinals=OrdinalsDict({n.id: o for o, n in ordinals_neurons}),
            neurons=[n for _, n in ordinals_neurons],
        )


class _LayerNeuronsList(_NeuronsList, LayerNeurons):
    def __init__(
        self,
        network: "NeuralNetworkDefinitionDict",
        ordinals: Ordinals,
        neurons: Collection[Neuron],
        layer: LayerDefinition,
    ) -> None:
        super().__init__(network, ordinals, neurons)
        self._layer = layer

    @property
    def layer(self) -> LayerDefinition:
        return self._layer


class NeuralNetworkDefinitionDict(BaseNeuralNetworkDefinition):
    def __init__(
        self,
        layers: Iterable[LayerDefinition],
        neurons: Sequence[Sequence[Neuron]] | Mapping[int, Collection[Neuron]],
        ordinals: Ordinals | None = None,
        ordinals_per_layer: Sequence[Ordinals] | Mapping[int, Ordinals] | None = None,
    ) -> None:
        self._layers = LayerDefinitionsImpl.from_iter(layers)

        self._neurons_per_layer: Mapping[int, Collection[Neuron]]
        if isinstance(neurons, Sequence):
            self._neurons_per_layer = {ld.id: ns for ld, ns in zip(self._layers, neurons)}
        else:
            self._neurons_per_layer = neurons

        self._neurons: Mapping[int, Neuron] = {n.id: n for ns in self._neurons_per_layer.values() for n in ns}

        self._ordinals_per_layer: Mapping[int, Ordinals]
        if ordinals_per_layer is None:
            self._ordinals_per_layer = {
                ld.id: OrdinalsDict(
                    {n.id: LayerOrdinal(ld.id, o) for o, n in enumerate(self._neurons_per_layer[ld.id])}
                )
                for ld in self._layers
            }
        elif isinstance(ordinals_per_layer, Sequence):
            self._ordinals_per_layer = {ld.id: os for ld, os in zip(self._layers, ordinals_per_layer)}
        else:
            self._ordinals_per_layer = ordinals_per_layer

        self._ordinals: Ordinals
        if ordinals is None:
            self._ordinals = OrdinalsDict({i: o for os in self._ordinals_per_layer.values() for i, o in os.items()})
        else:
            self._ordinals = ordinals

        self._items = MapCollection(lambda ld: (ld, self[ld]), self.layers)

    @property
    def ordinals(self) -> Ordinals:
        return self._ordinals

    @cache
    def __getitem__(self, layer: int | LayerDefinition) -> LayerNeurons:
        layer_id = get_layer_id(layer)

        return _LayerNeuronsList(
            network=self,
            ordinals=self._ordinals_per_layer[layer_id],
            neurons=self._neurons_per_layer[layer_id],
            layer=self._layers[layer_id],
        )

    def __len__(self) -> int:
        return len(self._layers)

    @property
    def layers(self) -> LayerDefinitions:
        return self._layers

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
