import hashlib
from typing import Any, OrderedDict, Sequence, TypeVar

from lib.nn.sources.base import LayerOrdinal, Network
from lib.nn.sources.minimal_api.base import MinimalAPINetwork
from lib.nn.sources.minimal_api.ordinals import MinimalAPIOrdinals
from lib.nn.sources.minimal_api_bridge import NetworkImpl
from lib.nn.sources.minimal_api_bridge_reverse import MinimalAPINetworkFromFullProxy, MinimalAPIOrdinalsFromFullProxy
from lib.utils import Delegate, MapSequence, cache, delegate

_TNeurons = TypeVar("_TNeurons")


@delegate("delegate")
class _MinimalAPIMergeFactsView(MinimalAPINetwork[_TNeurons]):
    def __init__(self, network: MinimalAPINetwork[_TNeurons]) -> None:
        self.delegate = network
        self._id_to_hash_mapping: dict[int, Any] = {}
        self._hash_to_id_mapping: dict[Any, int] = {}
        self._id_mapping: dict[int, int] = {}

    get_layers = Delegate()
    get_layers_map = Delegate()
    get_inputs = Delegate()
    get_input_weights = Delegate()
    get_input_lengths = Delegate()
    get_biases = Delegate()
    get_values_numpy = Delegate()
    get_values_torch = Delegate()
    get_transformations = Delegate()
    slice = Delegate()
    select_ids = Delegate()

    def get_ids(self, neurons: _TNeurons) -> Sequence[int]:
        for layer in self.get_layers():
            if layer.type == "FactLayer":
                self.get_layer_neurons(layer.id)

        return MapSequence(lambda id: self._id_mapping.get(id, id), self.delegate.get_ids(neurons))

    @cache
    def get_layer_neurons(self, layer_id: int) -> _TNeurons:
        layer = self.get_layers_map()[layer_id]

        if layer.type != "FactLayer":
            return self.delegate.get_layer_neurons(layer_id)

        underlying_ids: list[int] = []

        neurons = self.delegate.get_layer_neurons(layer_id)
        for id, arr in zip(self.delegate.get_ids(neurons), self.delegate.get_values_numpy(neurons)):
            h = hashlib.sha256(arr.data, usedforsecurity=False).digest()
            self._id_to_hash_mapping[id] = h
            if h not in self._hash_to_id_mapping:
                self._hash_to_id_mapping[h] = id
                underlying_ids.append(id)
            self._id_mapping[id] = self._hash_to_id_mapping[h]

        return self.delegate.select_ids(neurons, underlying_ids)


@delegate("delegate")
class _MinimalAPIMergeFactsViewOrdinals(MinimalAPIOrdinals):
    def __init__(self, minimal_api: _MinimalAPIMergeFactsView, ordinals: MinimalAPIOrdinals) -> None:
        self.delegate = ordinals
        self._minimal_api = minimal_api

    get_ordinal = Delegate()
    get_all_ordinals = Delegate()
    get_id = Delegate()

    @cache
    def get_ordinals_for_layer(self, layer_id: int) -> OrderedDict[int, LayerOrdinal]:
        layer = self._minimal_api.get_layers_map()[layer_id]
        if layer.type != "FactLayer":
            return self.delegate.get_ordinals_for_layer(layer_id)

        neurons = self._minimal_api.get_layer_neurons(layer_id)
        ids = self._minimal_api.get_ids(neurons)
        return OrderedDict(((id, LayerOrdinal(layer=layer_id, ordinal=o)) for o, id in enumerate(ids)))


class MergeFactsView(NetworkImpl):
    """A view of a given Network that merges all matching Fact neurons together."""

    def __init__(self, network: Network) -> None:
        minimal_api = _MinimalAPIMergeFactsView(MinimalAPINetworkFromFullProxy(network))
        minimal_ordinals = _MinimalAPIMergeFactsViewOrdinals(minimal_api, MinimalAPIOrdinalsFromFullProxy(network))

        super().__init__(minimal_api, custom_ordinals=minimal_ordinals)
