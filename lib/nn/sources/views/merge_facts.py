import hashlib
import itertools
from typing import Any, Callable, Collection, Iterable, Iterator

import numpy as np
import torch

from lib.nn.definitions.ops import TransformationDef
from lib.nn.sources.base_source import BaseLayerNeurons, BaseNeuralNetworkDefinition, BaseNeurons, BaseOrdinals
from lib.nn.sources.source import (
    LayerDefinition,
    LayerDefinitions,
    LayerNeurons,
    LayerOrdinal,
    NeuralNetworkDefinition,
    Neurons,
    Ordinals,
    WeightDefinition,
    get_layer_id,
)
from lib.nn.sources.utils import OrdinalsDict
from lib.utils import MapCollection, cache


class _MergeViewOrdinals(BaseOrdinals):
    def __init__(self, ordinals: Ordinals, ordinals_mapping: Callable[[LayerOrdinal], LayerOrdinal]) -> None:
        super().__init__()
        self._ordinals = ordinals
        self._mapping = ordinals_mapping

    def ids(self) -> Collection[int]:
        return self._ordinals.ids()

    def items(self) -> Collection[tuple[int, LayerOrdinal]]:
        return MapCollection(lambda id: (id, self.__getitem__(id)), self.ids())

    def values(self) -> Collection[LayerOrdinal]:
        return MapCollection(lambda id: self.__getitem__(id), self.ids())

    def __iter__(self) -> Iterator[LayerOrdinal]:
        yield from (self.__getitem__(id) for id in self.ids())

    def __len__(self) -> int:
        return len(self._ordinals)

    def __contains__(self, o: object) -> bool:
        return isinstance(o, LayerOrdinal) and o in self._ordinals and self._mapping(o) == o

    def __getitem__(self, id: int) -> LayerOrdinal:
        o = self._ordinals.__getitem__(id)
        return self._mapping(o)


class _MergeViewNeurons(BaseNeurons):
    def __init__(self, neurons: Neurons, ordinals: _MergeViewOrdinals) -> None:
        self._neurons = neurons
        self._ordinals = ordinals
        self._ordinal_mapping = ordinals._mapping

    @property
    def ordinals(self) -> Ordinals:
        return self._ordinals

    @property
    def inputs(self) -> "_MergeViewNeurons":
        inp = self._neurons.inputs
        return _MergeViewNeurons(neurons=inp, ordinals=_MergeViewOrdinals(inp.ordinals, self._ordinal_mapping))

    @property
    def ids(self) -> Collection[int]:
        return self._neurons.ids

    @property
    def input_lengths(self) -> Collection[int]:
        return self._neurons.input_lengths

    def __len__(self) -> int:
        return len(self._neurons)

    @property
    def input_weights(self) -> Iterable[WeightDefinition]:
        return self._neurons.input_weights

    @property
    def biases(self) -> Collection[WeightDefinition]:
        return self._neurons.biases

    def get_values_numpy(self) -> Collection[np.ndarray]:
        return self._neurons.get_values_numpy()

    def get_values_torch(self) -> Collection[torch.Tensor]:
        return self._neurons.get_values_torch()

    def get_transformations(self) -> Collection[TransformationDef | None]:
        return self._neurons.get_transformations()

    def gather(self, what: list[int | LayerOrdinal]) -> "_MergeViewNeurons":
        what = [self._ordinal_mapping(w) if isinstance(w, LayerOrdinal) else w for w in what]

        new_neurons = self._neurons.gather(what)
        return _MergeViewNeurons(
            neurons=new_neurons,
            ordinals=_MergeViewOrdinals(ordinals=new_neurons.ordinals, ordinals_mapping=self._ordinal_mapping),
        )


class _MergeViewLayerNeurons(_MergeViewNeurons, BaseLayerNeurons):
    def __init__(self, neurons: Neurons, ordinals: _MergeViewOrdinals, layer: LayerDefinition) -> None:
        super().__init__(neurons, ordinals)
        self._layer = layer

    @property
    def layer(self) -> LayerDefinition:
        return self._layer


class MergeFactsView(BaseNeuralNetworkDefinition):
    def __init__(self, network: NeuralNetworkDefinition) -> None:
        self.delegate = network

        self._hash_to_ordinal_mapping: dict[Any, LayerOrdinal] = {}
        self._ordinal_to_hash_mapping: dict[LayerOrdinal, Any] = {}
        self._mapping_underlying: dict[LayerOrdinal, LayerOrdinal] = {}

        self._layer_neurons: dict[int, Neurons] = {}
        self._layer_ordinals: dict[int, list[LayerOrdinal]] = {}

    def _map_ordinal(self, orig: LayerOrdinal) -> LayerOrdinal:
        self._update_mappings_for_layer(orig.layer)
        return self._mapping_underlying.get(orig, orig)

    @cache  # runs at most once for each layer thanks to the @cache decorator
    def _update_mappings_for_layer(self, layer_id: int):
        if self.layers[layer_id].type != "FactLayer":
            return

        ord_it = iter(itertools.count())

        def _next_new_ord():
            return LayerOrdinal(layer=layer_id, ordinal=next(ord_it))

        underlying_ids: list[int] = []
        underlying_ordinals: list[LayerOrdinal] = []

        neurons = self.delegate[layer_id]
        for id, o, arr in zip(neurons.ids, neurons.ordinals, neurons.get_values_numpy()):
            h = hashlib.sha256(arr.data, usedforsecurity=False).digest()
            self._ordinal_to_hash_mapping[o] = h
            if h not in self._hash_to_ordinal_mapping:
                self._hash_to_ordinal_mapping[h] = _next_new_ord()
                underlying_ids.append(id)
                underlying_ordinals.append(o)
            self._mapping_underlying[o] = self._hash_to_ordinal_mapping[h]

        self._layer_neurons[layer_id] = neurons.gather(underlying_ids)
        self._layer_ordinals[layer_id] = underlying_ordinals

    @property
    @cache
    def ordinals(self) -> Ordinals:
        return _MergeViewOrdinals(self.delegate.ordinals, self._map_ordinal)

    def __getitem__(self, layer: int | LayerDefinition) -> LayerNeurons:
        layer_id = get_layer_id(layer)
        self._update_mappings_for_layer(layer_id)

        layer = self.layers[layer_id]

        if layer_id in self._layer_neurons:
            neurons = self._layer_neurons[layer_id]
        else:
            neurons = self.delegate[layer_id]

        if layer_id in self._layer_ordinals:
            ordinals = OrdinalsDict({id: o for id, o in zip(neurons.ids, self._layer_ordinals[layer_id])})
        else:
            ordinals = self.delegate[layer_id].ordinals

        return _MergeViewLayerNeurons(
            neurons=neurons,
            ordinals=_MergeViewOrdinals(ordinals, self._map_ordinal),
            layer=layer,
        )

    def __len__(self) -> int:
        return len(self.delegate)

    @property
    def layers(self) -> LayerDefinitions:
        return self.delegate.layers

    def items(self) -> Collection[tuple[LayerDefinition, LayerNeurons]]:
        return MapCollection(lambda k: (k, self[k]), self.layers)

    def __iter__(self) -> Iterator[LayerNeurons]:
        yield from (self[ld] for ld in self.layers)

    def __contains__(self, o: object) -> bool:
        if isinstance(o, (LayerDefinition, int)):
            return o in self.layers

        if isinstance(o, LayerOrdinal):
            return o in self.ordinals

        raise ValueError()
