from typing import Callable, Sequence

import numpy as np

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation
from lib.vectorize.pipeline.utils.ref_groups import (
    GatherRefsTransform,
    build_grouper_for_aggregate,
    get_refs,
    remap_refs,
)
from lib.vectorize.pipeline.utils.values import HashableArray

ForwardPassRunner = Callable[[VectorizedOpSeqNetwork], dict[int, dict[str, np.ndarray]]]


class IsoCompression(LayerwiseOperation):
    def __init__(
        self,
        network: VectorizedLayerNetwork,
        tail: Callable[[VectorizedLayerNetwork], VectorizedOpSeqNetwork],
        forward_pass_runner: ForwardPassRunner,
        debug_print: bool,
        n_repeats: int = 4,
    ) -> None:
        self.network = network
        self.tail = tail
        self.forward_pass_runner = forward_pass_runner
        self.debug_print = debug_print
        self._counts = ComputeLayerCounts(network)
        self._n_repeats = n_repeats

        self._all_values: dict[int, dict[str, np.ndarray]]
        self._final_layer_ids: dict[int, str]

    def _get_layer_ord_map(
        self,
        values: np.ndarray,
    ) -> tuple[dict[int, int], list[int]]:
        value_to_key_map: dict[HashableArray, int] = {}
        out_ord_map: dict[int, int] = {}
        out_gather: list[int] = []

        for i, value_pair in enumerate(values):
            value_pair = HashableArray(value_pair)
            key = value_to_key_map.get(value_pair, None)

            if key is None:
                key = len(out_gather)
                value_to_key_map[value_pair] = key
                out_gather.append(i)

            if i != key:
                out_ord_map[i] = key

        return out_ord_map, out_gather

    def _remap_input(self, batch_id: int, layer_id: str, transform: GatherRefsTransform, input: Input):
        refs = get_refs(self._counts, batch_id, input)
        if self.debug_print:
            print(f"ISO COMPRESSION: {layer_id}: Remapping {len(refs)} to {len(transform.ordinals)}.")
        out = remap_refs(self._counts, batch_id, input, transform)
        assert out

    def _get_new_aggregate(self, aggregate: Reduce, refs_groups: Sequence[tuple]) -> Reduce:
        match aggregate:
            case Noop():
                return aggregate
            case FixedCountReduce():
                return aggregate
            case UnevenReduce(reduce=r):
                return UnevenReduce([len(chunk) for chunk in refs_groups], reduce=r)
            case _:
                assert False, f"{aggregate}"

    def _remap_layer(self, batch_id: int, layer_id: str, gather: list[int], layer: Layer):
        grouper = build_grouper_for_aggregate(layer.aggregate)
        transform = GatherRefsTransform(grouper, gather)

        match layer.base:
            case InputLayerBase(input=input):
                self._remap_input(batch_id, layer_id, transform, input)
            case LinearLayerBase(input=input, weight=weight) | LinearGatherLayerBase(input=input, weight=weight):
                self._remap_input(batch_id, layer_id, transform, input)
                self._remap_input(batch_id, layer_id, transform, weight)

        layer.aggregate = self._get_new_aggregate(layer.aggregate, transform.last_groups)

    def _reinitialize_weights(self, rng: np.random.Generator, net: VectorizedOpSeqNetwork):
        for weight in net.weights.values():
            weight.value = rng.uniform(-10, 10, size=weight.value.shape)

    def _before_all(self):
        op_seq_network = self.tail(self.network)

        self._final_layer_ids = {}
        for batch_id, batch in self.network.batches.items():
            self._final_layer_ids[batch_id] = next(iter(reversed(batch.layers)))

        _all_values: list[dict[int, dict[str, np.ndarray]]] = []
        assert self._n_repeats >= 1
        rng = np.random.default_rng()
        for i in range(self._n_repeats):
            if i >= 1:
                self._reinitialize_weights(rng, op_seq_network)
            _this_values = self.forward_pass_runner(op_seq_network)
            _all_values.append(_this_values)

        self._all_values = {}
        for batch_id in _all_values[0]:
            self._all_values[batch_id] = {}

            for layer_id in _all_values[0][batch_id]:
                self._all_values[batch_id][layer_id] = np.stack([vs[batch_id][layer_id] for vs in _all_values], axis=1)

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if layer_id == self._final_layer_ids[batch]:
            return layer

        batch_values = self._all_values[batch]

        try:
            values = batch_values["l_" + layer_id]
        except KeyError:
            return layer

        ord_map, gather = self._get_layer_ord_map(values)
        self._remap_layer(batch, layer_id, gather, layer)
        layer.ord_map = ord_map

        return layer


def build_iso_compression_factory(
    tail: Callable[[VectorizedLayerNetwork], VectorizedOpSeqNetwork],
    forward_pass_runner: Callable[[VectorizedOpSeqNetwork], dict[int, dict[str, np.ndarray]]],
    debug_print: bool,
):
    def _f(network: VectorizedLayerNetwork):
        return IsoCompression(network, tail, forward_pass_runner, debug_print=debug_print)

    return _f
