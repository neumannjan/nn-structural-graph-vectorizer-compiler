import hashlib
from typing import Any, OrderedDict, Sequence, TypeVar

from lib.nn.sources.base import LayerOrdinal, Network
from lib.nn.sources.minimal_api.base import MinimalAPINetwork
from lib.nn.sources.minimal_api.ordinals import MinimalAPIOrdinals
from lib.nn.sources.minimal_api_bridge import NetworkImpl
from lib.nn.sources.minimal_api_bridge_reverse import MinimalAPINetworkFromFullProxy, MinimalAPIOrdinalsFromFullProxy
from lib.utils import DelegatedMethod, MapSequence, cache, delegate

_TNeurons = TypeVar("_TNeurons")


@delegate
class _MinimalAPIMergeFactsView(MinimalAPINetwork[_TNeurons], MinimalAPIOrdinals):
    def __init__(self, network: MinimalAPINetwork[_TNeurons], ordinals: MinimalAPIOrdinals) -> None:
        self.network = network
        self.ordinals = ordinals
        self._id_to_hash_mapping: dict[int, Any] = {}
        self._hash_to_id_mapping: dict[Any, int] = {}
        self._id_mapping: dict[int, int] = {}

    get_layers = DelegatedMethod("network")
    get_layers_map = DelegatedMethod("network")
    get_inputs = DelegatedMethod("network")
    get_input_weights = DelegatedMethod("network")
    get_input_lengths = DelegatedMethod("network")
    get_biases = DelegatedMethod("network")
    get_values_numpy = DelegatedMethod("network")
    get_values_torch = DelegatedMethod("network")
    get_transformations = DelegatedMethod("network")
    get_aggregations = DelegatedMethod("network")
    slice = DelegatedMethod("network")
    select_ids = DelegatedMethod("network")

    get_ordinal = DelegatedMethod("ordinals")
    get_all_ordinals = DelegatedMethod("ordinals")
    get_id = DelegatedMethod("ordinals")

    def get_ids(self, neurons: _TNeurons) -> Sequence[int]:
        for layer in self.get_layers():
            if layer.type == "FactLayer":
                self.get_layer_neurons(layer.id)

        return MapSequence(lambda id: self._id_mapping.get(id, id), self.network.get_ids(neurons))

    @cache
    def get_layer_neurons(self, layer_id: int) -> _TNeurons:
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


class MergeFactsView(NetworkImpl):
    """A view of a given Network that merges all matching Fact neurons together."""

    def __init__(self, network: Network) -> None:
        minimal_api = _MinimalAPIMergeFactsView(
            MinimalAPINetworkFromFullProxy(network), MinimalAPIOrdinalsFromFullProxy(network)
        )

        super().__init__(minimal_api, custom_ordinals=minimal_api)
