import hashlib
from typing import Any, OrderedDict, Sequence

from lib.nn.sources.base import LayerOrdinal
from lib.nn.sources.views.view import (
    View,
    ViewBasis,
    view,
    view_basis,
)
from lib.utils import MapSequence, cache


@view_basis
class MergeFactsViewBasis(ViewBasis):
    def __init__(self, network, ordinals) -> None:
        super().__init__(network, ordinals)
        self._id_to_hash_mapping: dict[int, Any] = {}
        self._hash_to_id_mapping: dict[Any, int] = {}
        self._id_mapping: dict[int, int] = {}

    def get_ids(self, neurons) -> Sequence[int]:
        for layer in self.get_layers():
            if layer.type == "FactLayer":
                self.get_layer_neurons(layer.id)

        return MapSequence(lambda id: self._id_mapping.get(id, id), self.network.get_ids(neurons))

    @cache
    def get_layer_neurons(self, layer_id: int):
        layers_map = self.get_layers_map()
        layer = layers_map[layer_id]

        if layer.type != "FactLayer":
            return self.network.get_layer_neurons(layer_id)

        underlying_ids: list[int] = []

        neurons = self.network.get_layer_neurons(layer_id)
        for id, arr in zip(self.network.get_ids(neurons), self.network.get_values_numpy(neurons)):
            h = hashlib.sha256(arr.data, usedforsecurity=False).digest()
            self._id_to_hash_mapping[id] = h
            if h not in self._hash_to_id_mapping:
                self._hash_to_id_mapping[h] = id
                underlying_ids.append(id)
            self._id_mapping[id] = self._hash_to_id_mapping[h]

        return self.network.select_ids(neurons, underlying_ids)

    @cache
    def get_ordinals_for_layer(self, layer_id: int) -> OrderedDict[int, LayerOrdinal]:
        layer = self.get_layers_map()[layer_id]
        if layer.type != "FactLayer":
            return self.ordinals.get_ordinals_for_layer(layer_id)

        neurons = self.get_layer_neurons(layer_id)
        ids = self.get_ids(neurons)
        return OrderedDict(((id, LayerOrdinal(layer=layer_id, ordinal=o)) for o, id in enumerate(ids)))


@view(MergeFactsViewBasis)
class MergeFactsView(View):
    """A view of a given Network that merges all matching Fact neurons together."""

    pass
