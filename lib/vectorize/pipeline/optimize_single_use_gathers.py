import itertools
from typing import Iterable, Iterator, Literal, TypeVar

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.utils.chain_graph import ComputeChainGraph
from lib.vectorize.pipeline.utils.ref_groups import get_ref_groups


def _reorder_aggregate(aggregate: Reduce, ordinals: list[int]):
    match aggregate:
        case Noop():
            pass
        case FixedCountReduce():
            pass
        case UnevenReduce(counts=counts):
            aggregate.counts = [counts[i] for i in ordinals]


_T = TypeVar("_T")


def iter_lookahead(it: Iterable[_T]) -> Iterable[tuple[_T, _T | None]]:
    if not isinstance(it, Iterator):
        it = iter(it)

    try:
        v = next(it)
    except StopIteration:
        return

    try:
        while True:
            v_next = next(it)
            yield v, v_next
            v = v_next
    except StopIteration:
        yield v, None


class OptimizeSingleUseGathers:
    def __init__(self, network: VectorizedLayerNetwork, max_chain_length: int, debug: bool) -> None:
        self.network = network
        self.max_chain_length = max_chain_length
        self.debug = debug

        # TODO support facts also!
        self._compute_chain_graph = ComputeChainGraph(network)
        self._compute_counts = ComputeLayerCounts(network)

    def _reorder_layer_output(
        self, batch: int, layer: Layer, ordinals2: list[int], chain_i: int, upcoming_layer_id: str | None
    ) -> Literal["ignore", "found", "not_found", "inapplicable"]:
        match layer:
            case Layer(
                base=LinearLayerBase(
                    input=GatheredLayers(gather=GenericGather() as gather_a),
                    weight=GatheredLayers(gather=GenericGather() as gather_b),
                ),
                aggregate=aggregate,
            ) if len(gather_a.ordinals) == len(gather_b.ordinals):
                first_ord_groups = dict(enumerate(get_ref_groups(aggregate, gather_a.ordinals)))
                gather_a_ordinals_new = [v for b in ordinals2 for v in first_ord_groups[b]]

                if len(gather_a_ordinals_new) > len(gather_a.ordinals) and chain_i >= self.max_chain_length:
                    return "ignore"

                first_ord_groups = dict(enumerate(get_ref_groups(aggregate, gather_b.ordinals)))
                gather_b.ordinals = [v for b in ordinals2 for v in first_ord_groups[b]]

                _reorder_aggregate(aggregate, ordinals2)
                return "found"
            case Layer(
                base=(
                    LinearLayerBase(
                        input=GatheredLayers(gather=GenericGather() as gather),
                        weight=GatheredLayers() as other,
                    )
                    | LinearLayerBase(
                        input=GatheredLayers() as other,
                        weight=GatheredLayers(gather=GenericGather() as gather),
                    )
                ),
                aggregate=FixedCountReduce(period=period) as aggregate,
            ) if self._compute_counts.compute_input_count(batch, other) == period:
                first_ord_groups = dict(enumerate(get_ref_groups(aggregate, gather.ordinals)))
                gather_ordinals_new = [v for b in ordinals2 for v in first_ord_groups[b]]

                if len(gather_ordinals_new) > len(gather.ordinals) and chain_i >= self.max_chain_length:
                    return "ignore"

                gather.ordinals = gather_ordinals_new
                return "found"
            case Layer(
                base=(
                    InputLayerBase(input=GatheredLayers(gather=GenericGather() as gather))
                    | LinearGatherLayerBase(gather=GenericGather() as gather)
                ),
                aggregate=aggregate,
            ):
                first_ord_groups = dict(enumerate(get_ref_groups(aggregate, gather.ordinals)))

                gather_ordinals_new = [v for b in ordinals2 for v in first_ord_groups[b]]

                if len(gather_ordinals_new) > len(gather.ordinals) and chain_i >= self.max_chain_length:
                    return "ignore"

                gather.ordinals = gather_ordinals_new
                _reorder_aggregate(aggregate, ordinals2)
                return "found"
            case Layer(base=InputLayerBase(input=GatheredLayers(refs=[_], gather=NoopGather()))):
                return "not_found"
            case Layer(
                base=LinearLayerBase(
                    input=GatheredLayers(refs=[(LayerRefs.TYPE_LAYER, layer_id2)], gather=NoopGather())
                )
            ) if upcoming_layer_id == layer_id2:
                return "not_found"
            case Layer(
                base=LinearLayerBase(
                    weight=GatheredLayers(refs=[(LayerRefs.TYPE_LAYER, layer_id2)], gather=NoopGather())
                )
            ) if upcoming_layer_id == layer_id2:
                return "not_found"
            case Layer(
                base=LinearGatherLayerBase(
                    input=GatheredLayers(refs=[(LayerRefs.TYPE_LAYER, layer_id2)], gather=NoopGather()),
                    gather=NoopGather(),
                )
            ) if upcoming_layer_id == layer_id2:
                return "not_found"
            case Layer(
                base=LinearGatherLayerBase(
                    weight=GatheredLayers(refs=[(LayerRefs.TYPE_LAYER, layer_id2)], gather=NoopGather()),
                    gather=NoopGather(),
                )
            ) if upcoming_layer_id == layer_id2:
                return "not_found"

        return "inapplicable"

    def _for_layers(self, batch: int, layers: dict[str, Layer]):
        g = self._compute_chain_graph(layers)

        for chain in g.iter_chains_with_ref_gathers(g):
            chain = list(chain)
            chain.reverse()
            i = 0
            old_ref_to_layer = None
            for (layer_id, new_ref_to_layer), _upcoming in iter_lookahead(chain):
                upcoming_layer_id = _upcoming[0] if _upcoming is not None else None
                match new_ref_to_layer:
                    case None | GatheredLayers(gather=NoopGather()):
                        ref_to_layer = old_ref_to_layer
                    case _:
                        ref_to_layer = new_ref_to_layer

                if self.debug:
                    print("OPTIMIZE_SINGLE_USE_GATHERS", layer_id, layers[layer_id], ref_to_layer)

                match ref_to_layer:
                    case GatheredLayers(gather=GenericGather(ordinals)):
                        match self._reorder_layer_output(
                            batch, layers[layer_id], ordinals, chain_i=i, upcoming_layer_id=upcoming_layer_id
                        ):
                            case "found":
                                ref_to_layer.gather = NoopGather()
                                if self.debug:
                                    print("OPTIMIZE_SINGLE_USE_GATHERS: CHANGED", layers[layer_id])
                                i += 1
                                old_ref_to_layer = None
                            case "ignore":
                                old_ref_to_layer = None
                            case "inapplicable":
                                old_ref_to_layer = None
                            case "not_found":
                                old_ref_to_layer = ref_to_layer
                    case _:
                        old_ref_to_layer = ref_to_layer

    def optimize_single_use_gathers(self):
        for batch_id, batch in self.network.batches.items():
            self._for_layers(batch_id, batch.layers)


def build_optimize_single_use_gathers(max_chain_length: int, debug: bool):
    def optimize_single_use_gathers(network: VectorizedLayerNetwork):
        OptimizeSingleUseGathers(network, max_chain_length=max_chain_length, debug=debug).optimize_single_use_gathers()
        return network

    return optimize_single_use_gathers
