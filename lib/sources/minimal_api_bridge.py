from typing import Collection, Generic, Iterable, Iterator, Sequence

import numpy as np
import torch

from lib.nn.definitions.ops import AggregationDef, TransformationDef
from lib.sources.base import LayerDefinition, LayerOrdinal, Ordinals, WeightDefinition, get_layer_id
from lib.sources.base_impl import BaseLayerNeurons, BaseNetwork, BaseNeurons, BaseOrdinals
from lib.sources.minimal_api.base import MinimalAPINetwork, TNeurons
from lib.sources.minimal_api.ordinals import MinimalAPIOrdinals, MinimalAPIOrdinalsImpl
from lib.sources.utils import LayerDefinitionsImpl
from lib.utils import MapCollection, cache


class _Ordinals(BaseOrdinals):
    def __init__(self, minimal_ordinals: MinimalAPIOrdinals, ids: Collection[int], layer_id: int | None) -> None:
        self._minimal_ordinals = minimal_ordinals
        self._ids = ids
        self._layer_id = layer_id

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, id: int) -> LayerOrdinal:
        return self._minimal_ordinals.get_ordinal(self._layer_id, id)

    def ids(self) -> Collection[int]:
        return self._ids

    def items(self) -> Collection[tuple[int, LayerOrdinal]]:
        return MapCollection(lambda id: (id, self[id]), self._ids)

    def values(self) -> Collection[LayerOrdinal]:
        return MapCollection(lambda id: self[id], self._ids)

    def __iter__(self) -> Iterator[LayerOrdinal]:
        return iter(self.values())

    @property
    @cache
    def _ids_set(self) -> set[int]:
        return set(self._ids)

    def __contains__(self, o: object) -> bool:
        if isinstance(o, LayerOrdinal):
            return self._minimal_ordinals.get_id(o) in self._ids_set

        if isinstance(o, int):
            return o in self._ids_set

        raise ValueError(f"Cannot check if value is in ordinals for value {o} of type {type(o)}.")


class _Neurons(BaseNeurons, Generic[TNeurons]):
    def __init__(
        self,
        minimal_api: MinimalAPINetwork[TNeurons],
        minimal_ordinals: MinimalAPIOrdinals,
        neurons: TNeurons,
        layer_id: int | None,
    ) -> None:
        self._minimal_api = minimal_api
        self._minimal_ordinals = minimal_ordinals
        self._neurons = neurons
        self._layer_id = layer_id

    def _map_neurons(self, new_neurons: TNeurons) -> "_Neurons":
        return _Neurons(self._minimal_api, self._minimal_ordinals, neurons=new_neurons, layer_id=None)

    @property
    @cache
    def ordinals(self) -> Ordinals:
        return _Ordinals(self._minimal_ordinals, self.ids, layer_id=self._layer_id)

    @property
    def ids(self) -> Sequence[int]:
        return self._minimal_api.get_ids(self._neurons)

    @property
    def input_lengths(self) -> Sequence[int]:
        return self._minimal_api.get_input_lengths(self._neurons)

    @property
    def inputs(self) -> "_Neurons":
        input_neurons = self._minimal_api.get_inputs(self._neurons)
        return self._map_neurons(input_neurons)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def input_weights(self) -> Iterable[WeightDefinition]:
        return self._minimal_api.get_input_weights(self._neurons)

    @property
    def biases(self) -> Sequence[WeightDefinition]:
        return self._minimal_api.get_biases(self._neurons)

    def get_values_numpy(self) -> Collection[np.ndarray]:
        return self._minimal_api.get_values_numpy(self._neurons)

    def get_values_torch(self) -> Collection[torch.Tensor]:
        return self._minimal_api.get_values_torch(self._neurons)

    def get_transformations(self) -> Sequence[TransformationDef | None]:
        return self._minimal_api.get_transformations(self._neurons)

    def get_aggregations(self) -> Sequence[AggregationDef | None]:
        return self._minimal_api.get_aggregations(self._neurons)

    def slice(self, sl: slice) -> "_Neurons":
        neurons = self._minimal_api.slice(self._neurons, sl)
        return self._map_neurons(neurons)

    def select_ids(self, ids: Sequence[int]) -> "_Neurons":
        neurons = self._minimal_api.select_ids(self._neurons, ids)
        return self._map_neurons(neurons)

    def select_ord(self, ords: Sequence[LayerOrdinal]) -> "_Neurons":
        ids = [self._minimal_ordinals.get_id(o) for o in ords]
        return self.select_ids(ids)


class _LayerNeurons(_Neurons, BaseLayerNeurons):
    def __init__(
        self,
        minimal_api: MinimalAPINetwork[TNeurons],
        minimal_ordinals: MinimalAPIOrdinals,
        neurons: TNeurons,
        layer_id: int,
    ) -> None:
        super().__init__(minimal_api, minimal_ordinals, neurons, layer_id)
        assert self._layer_id is not None
        self._layer = self._minimal_api.get_layers_map()[self._layer_id]

    @property
    def layer(self) -> LayerDefinition:
        return self._layer


class NetworkImpl(BaseNetwork):
    """A wrapper implementation of the 'full' `Network` API, based on an provided minimal API implementation."""

    def __init__(self, minimal_api: MinimalAPINetwork, custom_ordinals: MinimalAPIOrdinals | None = None) -> None:
        self._minimal_api = minimal_api

        if custom_ordinals is None:
            self._minimal_ordinals = MinimalAPIOrdinalsImpl(minimal_api)
        else:
            self._minimal_ordinals = custom_ordinals

    @property
    @cache
    def ordinals(self) -> Ordinals:
        return _Ordinals(
            minimal_ordinals=self._minimal_ordinals,
            ids=self._minimal_ordinals.get_all_ordinals().keys(),
            layer_id=None,
        )

    @cache
    def __getitem__(self, layer: int | LayerDefinition) -> _LayerNeurons:
        layer_id = get_layer_id(layer)

        return _LayerNeurons(
            minimal_api=self._minimal_api,
            minimal_ordinals=self._minimal_ordinals,
            neurons=self._minimal_api.get_layer_neurons(layer_id),
            layer_id=layer_id,
        )

    def __len__(self) -> int:
        return len(self._minimal_ordinals.get_all_ordinals())

    @property
    @cache
    def layers(self) -> LayerDefinitionsImpl:
        return LayerDefinitionsImpl.from_iter(self._minimal_api.get_layers())

    def items(self) -> Collection[tuple[LayerDefinition, _LayerNeurons]]:
        return MapCollection(lambda layer: (layer, self[layer]), self._minimal_api.get_layers())

    def __iter__(self) -> Iterator[_LayerNeurons]:
        yield from (self[layer] for layer in self._minimal_api.get_layers())

    def __contains__(self, o: object) -> bool:
        if isinstance(o, (LayerOrdinal, int)):
            return o in self.ordinals

        if isinstance(o, LayerDefinition):
            return o in self.layers

        raise ValueError()

    @property
    def minimal_api(self) -> MinimalAPINetwork | None:
        return self._minimal_api

    @property
    def minimal_api_ordinals(self) -> MinimalAPIOrdinals | None:
        return self._minimal_ordinals
