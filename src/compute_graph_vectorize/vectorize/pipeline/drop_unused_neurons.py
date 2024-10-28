import itertools

import numpy as np

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from compute_graph_vectorize.vectorize.pipeline.utils.ref_groups import SeqGrouper, build_grouper_for_aggregate


class DropUnusedNeuronsNoOrdRemap:
    def __init__(self, network: VectorizedLayerNetwork):
        self.network = network
        self._counts = ComputeLayerCounts(network)

    def _update_masks_for_refs(self, masks: dict[str, np.ndarray], refs: Refs):
        layers = set((l for t, l in zip(refs.types, refs.layer_ids) if t == Refs.TYPE_LAYER))

        for layer_id in layers:
            o = [o for t, l, o in refs if t == Refs.TYPE_LAYER and l == layer_id]
            masks[layer_id][o] = True

    def _update_masks_for_layer(self, masks: dict[str, np.ndarray], layer: Layer):
        match layer.base:
            case InputLayerBase(input=Refs() as input):
                self._update_masks_for_refs(masks, input)
            case (
                LinearLayerBase(input=Refs() as input, weight=Refs() as weight)
                | LinearGatherLayerBase(input=Refs() as input, weight=Refs() as weight)
            ):
                self._update_masks_for_refs(masks, input)
                self._update_masks_for_refs(masks, weight)
            case _:
                raise ValueError(layer.base)

    def _mask_to_ord_map(self, mask: np.ndarray) -> dict[int, int]:
        (vals_arr,) = np.where(mask)
        vals: list[int] = vals_arr.tolist()

        out = {k: i for i, k in enumerate(vals) if i != k}
        print(len(mask), "->", np.sum(mask), out)
        return out

    def _drop_by_mask_in_aggregate(self, aggregate: Reduce, mask: list[bool]) -> None:
        match aggregate:
            case UnevenReduce(counts=counts):
                counts = [c for m, c in zip(mask, counts) if m]
                aggregate.counts = counts
            case FixedCountReduce():
                pass
            case Noop():
                pass
            case _:
                raise ValueError(aggregate)

    def _drop_by_mask_in_refs(self, refs: Refs, mask: list[bool], grouper: SeqGrouper) -> Refs:
        groups = grouper.group(refs)
        groups = [g for m, g in zip(mask, groups) if m]
        new_refs = Refs.from_iter(grouper.ungroup(groups))
        return new_refs

    def _drop_by_mask_in_layer(self, batch: int, layer: Layer, mask: list[bool]) -> None:
        grouper = build_grouper_for_aggregate(layer.aggregate)
        self._drop_by_mask_in_aggregate(layer.aggregate, mask)

        match layer.base:
            case InputLayerBase(input=Refs() as input) as base:
                base.input = self._drop_by_mask_in_refs(input, mask, grouper)
            case LinearLayerBase(input=Refs() as input, weight=Refs() as weight) as base:
                base.input = self._drop_by_mask_in_refs(input, mask, grouper)
                base.weight = self._drop_by_mask_in_refs(weight, mask, grouper)
            case _:
                raise ValueError()

        layer.count = self._counts.compute_layer_count(batch, layer)

    def __call__(self):
        for batch_id, batch in self.network.batches.items():
            batch_masks = {}

            for layer_id, layer in batch.layers.items():
                assert layer.count is not None
                assert not layer.ord_map
                batch_masks[layer_id] = np.zeros(layer.count, dtype=bool)

            for layer_id, layer in batch.layers.items():
                self._update_masks_for_layer(batch_masks, layer)

            for layer_id, layer in itertools.islice(batch.layers.items(), len(batch.layers) - 1):
                layer.ord_map = self._mask_to_ord_map(batch_masks[layer_id])

                self._drop_by_mask_in_layer(batch_id, layer, batch_masks[layer_id].tolist())


def drop_unused_neurons_no_ord_remap(network: VectorizedLayerNetwork):
    DropUnusedNeuronsNoOrdRemap(network)()
    return network
