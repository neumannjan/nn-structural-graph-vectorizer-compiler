from collections.abc import Collection, Sequence
from typing import Iterable, Mapping, OrderedDict, overload

import numpy as np
import torch

from lib.nn.definitions.ops import AggregationDef, TransformationDef
from lib.nn.sources.base import LayerDefinition, LayerOrdinal, Network, Neurons, WeightDefinition
from lib.nn.sources.minimal_api.base import MinimalAPINetwork
from lib.nn.sources.minimal_api.ordinals import MinimalAPIOrdinals
from lib.utils import cache


class MinimalAPINetworkFromFullProxy(MinimalAPINetwork[Neurons]):
    """A minimal API network proxy for an underlying full API `Network` implementation."""

    def __new__(cls, network: Network):
        if network.minimal_api is not None:
            return network.minimal_api

        return super().__new__(cls)

    def __init__(self, network: Network) -> None:
        self.delegate = network

    def get_layers(self) -> Sequence[LayerDefinition]:
        return self.delegate.layers.as_list()

    def get_layers_map(self) -> Mapping[int, LayerDefinition]:
        return self.delegate.layers.as_dict()

    def get_layer_neurons(self, layer_id: int) -> Neurons:
        return self.delegate[layer_id]

    def get_ids(self, neurons: Neurons) -> Sequence[int]:
        return neurons.ids

    def get_inputs(self, neurons: Neurons) -> Neurons:
        return neurons.inputs

    def get_input_lengths(self, neurons: Neurons) -> Sequence[int]:
        return neurons.input_lengths

    def get_input_weights(self, neurons: Neurons) -> Iterable[WeightDefinition]:
        return neurons.input_weights

    def get_biases(self, neurons: Neurons) -> Sequence[WeightDefinition]:
        return neurons.biases

    def get_values_numpy(self, neurons: Neurons) -> Collection[np.ndarray]:
        return neurons.get_values_numpy()

    def get_values_torch(self, neurons: Neurons) -> Collection[torch.Tensor]:
        return neurons.get_values_torch()

    def get_transformations(self, neurons: Neurons) -> Sequence[TransformationDef | None]:
        return neurons.get_transformations()

    def get_aggregations(self, neurons: Neurons) -> Sequence[AggregationDef | None]:
        return neurons.get_aggregations()

    def slice(self, neurons: Neurons, sl: slice) -> Neurons:
        return neurons.slice(sl)

    def select_ids(self, neurons: Neurons, ids: Sequence[int]) -> Neurons:
        return neurons.select_ids(ids)


class MinimalAPIOrdinalsFromFullProxy(MinimalAPIOrdinals):
    """A minimal API ordinals proxy for an underlying full API `Network` implementation."""

    def __new__(cls, network: Network):
        if network.minimal_api_ordinals is not None:
            return network.minimal_api_ordinals

        return super().__new__(cls)

    def __init__(self, network: Network) -> None:
        self.delegate = network

    @overload
    def get_ordinal(self, layer: int | None, id: int, /) -> LayerOrdinal:
        ...

    @overload
    def get_ordinal(self, id: int, /) -> LayerOrdinal:
        ...

    def get_ordinal(self, *kargs: int | None) -> LayerOrdinal:
        if len(kargs) == 1:
            layer = None
            (id,) = kargs
        elif len(kargs) == 2:
            layer, id = kargs
        else:
            raise RuntimeError()

        assert id is not None
        if layer is not None:
            return self.delegate[layer].ordinals[id]

        return self.delegate.ordinals[id]

    def get_id(self, o: LayerOrdinal) -> int:
        return self.delegate[o.layer].ids[o.ordinal]

    @cache
    def get_ordinals_for_layer(self, layer_id: int) -> OrderedDict[int, LayerOrdinal]:
        return OrderedDict(self.delegate[layer_id].ordinals.items())

    @cache
    def get_all_ordinals(self) -> dict[int, LayerOrdinal]:
        return dict(self.delegate.ordinals.items())
