from collections.abc import Sequence

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from compute_graph_vectorize.vectorize.pipeline.layerwise import LayerwiseOperation
from compute_graph_vectorize.vectorize.pipeline.utils.gather import combine_ord_maps_
from compute_graph_vectorize.vectorize.pipeline.utils.ref_groups import (
    CustomRefsTransform,
    GrouperError,
    SimpleOrderRefsTransform,
    SimpleUniqueRefsTransform,
    apply_refs_to_target,
    build_grouper_for_aggregate,
    get_remapped_refs,
)


class RemapOrdinals(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _for_refs(self, batch: Batch, refs: Refs):
        for i, (t, l, o) in enumerate(refs):
            if t != Refs.TYPE_LAYER:
                continue

            o_real = batch.layers[l].ord_map.get(o, o)
            if o_real != o:
                refs.ordinals[i] = o_real

    def __call__(self, batch_id: int, layer_id: str, layer: Layer) -> Layer:
        batch = self.network.batches[batch_id]

        match layer.base:
            case InputLayerBase(input=Refs() as input):
                self._for_refs(batch, input)
            case LinearLayerBase(input=Refs() as input, weight=Refs() as weight):
                self._for_refs(batch, input)
                self._for_refs(batch, weight)
            case LinearGatherLayerBase(input=Refs() as input, weight=Refs() as weight):
                self._for_refs(batch, input)
                self._for_refs(batch, weight)
            case _:
                assert False, f"{layer.base}"
        return layer

    def _after_all(self):
        for batch in self.network.batches.values():
            for layer in batch.layers.values():
                layer.ord_map = {}


class OptimizeTailRefsToUniqueNoOrdRemap(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork, unique_margin_rate: float) -> None:
        self.network = network
        self.unique_margin_rate = unique_margin_rate
        self._final_layers = {batch: next(reversed(self.network.batches[batch].layers)) for batch in network.batches}
        self._counts = ComputeLayerCounts(network)

    def _get_new_aggregate(self, aggregate: Reduce, refs_chunks_uniq: Sequence[tuple]) -> Reduce:
        match aggregate:
            case Noop():
                return aggregate
            case FixedCountReduce():
                return aggregate
            case UnevenReduce(reduce=r):
                return UnevenReduce([len(chunk) for chunk in refs_chunks_uniq], reduce=r)
            case _:
                assert False, f"{aggregate}"

    def _apply_to_target(self, batch: int, layer: Layer, target: Refs | GenericGather | GatheredLayers):
        grouper = build_grouper_for_aggregate(layer.aggregate)

        transform1 = SimpleUniqueRefsTransform(grouper, with_ord_map=True)
        remapped_refs1 = get_remapped_refs(self._counts, batch, target, transform1)

        transform2 = SimpleOrderRefsTransform(grouper, with_ord_map=True)
        remapped_refs2 = get_remapped_refs(self._counts, batch, target, transform2)

        assert remapped_refs2 is not None

        if remapped_refs1 is None or 1 - len(remapped_refs1) / len(remapped_refs2) <= self.unique_margin_rate:
            transform, remapped_refs = transform2, remapped_refs2
        else:
            transform, remapped_refs = transform1, remapped_refs1

        apply_refs_to_target(remapped_refs, target)
        combine_ord_maps_(transform.ord_map, layer.ord_map)
        layer.ord_map = transform.ord_map
        layer.aggregate = self._get_new_aggregate(layer.aggregate, transform.last_groups)

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if layer_id == self._final_layers[batch]:
            # skip the final layer
            return layer

        match layer:
            case Layer(
                base=InputLayerBase(input=input),
            ):
                self._apply_to_target(batch, layer, input)
            case Layer(
                base=(LinearLayerBase() | LinearGatherLayerBase(gather=NoopGather())),
            ):
                # nothing to do here?
                pass
            case Layer(
                base=LinearGatherLayerBase(input=input, weight=weight, lifts=lifts, gather=GenericGather() as gather),
            ):
                # do we remove the gather altogether?
                linear_count = self._counts.compute_linear_count(batch, input, weight, lifts)
                if (
                    1 - len(gather.ordinals) / self._counts.compute_linear_count(batch, input, weight, lifts)
                    <= self.unique_margin_rate
                ):
                    grouper = build_grouper_for_aggregate(layer.aggregate)
                    transform = CustomRefsTransform(grouper, range(linear_count), with_ord_map=True)

                    success = get_remapped_refs(self._counts, batch, gather, transform) is not None

                    if success:
                        combine_ord_maps_(transform.ord_map, layer.ord_map)
                        layer.ord_map = transform.ord_map
                        layer.aggregate = self._get_new_aggregate(layer.aggregate, transform.last_groups)
                        layer.base = LinearLayerBase(input=input, weight=weight, lifts=lifts)

                        # can we repeat the tail application process now?
                        self(batch, layer_id, layer)

        return layer


def build_optimize_tail_refs_to_unique_no_ord_remap_factory(unique_margin_rate: float):
    def _f(network: VectorizedLayerNetwork):
        return OptimizeTailRefsToUniqueNoOrdRemap(network, unique_margin_rate)

    return _f
