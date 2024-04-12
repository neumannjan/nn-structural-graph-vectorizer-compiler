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
    def __init__(
        self, network: VectorizedLayerNetwork, max_chain_length: int | Literal["unlimited"], debug: bool
    ) -> None:
        self.network = network
        self.max_chain_length = float("inf") if max_chain_length == "unlimited" else max_chain_length
        self.debug = debug

        self._compute_chain_graph = ComputeChainGraph(network, types=(LayerRefs.TYPE_LAYER, LayerRefs.TYPE_FACT))
        self._compute_counts = ComputeLayerCounts(network)

        self._fact_reorders: dict[tuple[str, tuple[int, ...]], str] = {}

    def _reorder_fact(
        self, batch_id: int, orig_layer_id: str, fact_layer: FactLayer, ordinals2: list[int], chain_i: int
    ) -> tuple[Literal["found", "found_free"], str] | tuple[Literal["ignore"], None]:
        reordered_key = orig_layer_id, tuple(ordinals2)

        new_layer_id = self._fact_reorders.get(reordered_key, None)

        # We must create this version of the layer first.
        if new_layer_id is None:
            facts_new = [fact_layer.facts[o] for o in ordinals2]
            count_new = self._compute_counts.compute_facts_count(facts_new)
            new_layer = FactLayer(facts=facts_new, count=count_new, shape=fact_layer.shape)
        else:
            new_layer = self.network.fact_layers[new_layer_id]
            count_new = new_layer.count
            assert count_new is not None

        count_old = (
            self._compute_counts.compute_facts_count(fact_layer.facts) if fact_layer.count is None else fact_layer.count
        )

        is_free = count_new <= count_old

        # Let's not do it and use the original.
        if not is_free and chain_i >= self.max_chain_length:
            return "ignore", None

        if new_layer_id is None:
            new_layer_id = orig_layer_id + "__" + str(batch_id)

            self.network.fact_layers[new_layer_id] = new_layer
            self._fact_reorders[reordered_key] = new_layer_id

        return ("found_free", new_layer_id) if is_free else ("found", new_layer_id)

    def _reorder_layer_output(
        self, batch: int, layer: Layer, ordinals2: list[int], chain_i: int, upcoming_ref: tuple[int, str] | None
    ) -> Literal["ignore", "found_free", "found", "not_found", "inapplicable"]:
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

                is_free = len(gather_a_ordinals_new) <= len(gather_a.ordinals)

                if not is_free and chain_i >= self.max_chain_length:
                    return "ignore"

                first_ord_groups = dict(enumerate(get_ref_groups(aggregate, gather_b.ordinals)))
                gather_b.ordinals = [v for b in ordinals2 for v in first_ord_groups[b]]

                _reorder_aggregate(aggregate, ordinals2)
                return "found_free" if is_free else "found"
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

                is_free = len(gather_ordinals_new) <= len(gather.ordinals)

                if not is_free and chain_i >= self.max_chain_length:
                    return "ignore"

                gather.ordinals = gather_ordinals_new
                return "found_free" if is_free else "found"
            case Layer(
                base=(
                    InputLayerBase(input=GatheredLayers(gather=GenericGather() as gather))
                    | LinearGatherLayerBase(gather=GenericGather() as gather)
                ),
                aggregate=aggregate,
            ):
                first_ord_groups = dict(enumerate(get_ref_groups(aggregate, gather.ordinals)))

                gather_ordinals_new = [v for b in ordinals2 for v in first_ord_groups[b]]

                is_free = len(gather_ordinals_new) <= len(gather.ordinals)

                if not is_free and chain_i >= self.max_chain_length:
                    return "ignore"

                gather.ordinals = gather_ordinals_new
                _reorder_aggregate(aggregate, ordinals2)
                return "found_free" if is_free else "found"
            case Layer(base=InputLayerBase(input=GatheredLayers(refs=[_], gather=NoopGather()))):
                return "not_found"
            case Layer(
                base=LinearLayerBase(input=GatheredLayers(refs=[ref2], gather=NoopGather()))
            ) if upcoming_ref == ref2:
                return "not_found"
            case Layer(
                base=LinearLayerBase(weight=GatheredLayers(refs=[ref2], gather=NoopGather()))
            ) if upcoming_ref == ref2:
                return "not_found"
            case Layer(
                base=LinearGatherLayerBase(
                    input=GatheredLayers(refs=[ref2], gather=NoopGather()),
                    gather=NoopGather(),
                )
            ) if upcoming_ref == ref2:
                return "not_found"
            case Layer(
                base=LinearGatherLayerBase(
                    weight=GatheredLayers(refs=[ref2], gather=NoopGather()),
                    gather=NoopGather(),
                )
            ) if upcoming_ref == ref2:
                return "not_found"

        return "inapplicable"

    def _replace_references(self, gl: GatheredLayers, ref_from: tuple[int, str], id_to: str):
        for i, ref in enumerate(gl.refs):
            if ref == ref_from:
                gl.refs.layer_ids[i] = id_to

    def _for_layers(self, batch: int, layers: dict[str, Layer]):
        g = self._compute_chain_graph(layers)

        for chain in g.iter_chains_with_ref_gathers():
            chain = list(chain)
            chain.reverse()
            i = 0
            old_ref_to_layer = None
            for (ref, new_ref_to_layer), _upcoming in iter_lookahead(chain):
                upcoming_ref = _upcoming[0] if _upcoming is not None else None
                match new_ref_to_layer:
                    case None | GatheredLayers(gather=NoopGather()):
                        ref_to_layer = old_ref_to_layer
                    case _:
                        ref_to_layer = new_ref_to_layer

                t, l = ref

                if t == LayerRefs.TYPE_FACT:
                    value = self.network.fact_layers[l]
                elif t == LayerRefs.TYPE_LAYER:
                    value = layers[l]
                else:
                    assert False

                if self.debug:
                    print("OPTIMIZE_SINGLE_USE_GATHERS", ref, value, ref_to_layer, new_ref_to_layer)

                match ref_to_layer:
                    case GatheredLayers(gather=GenericGather(ordinals)):
                        if t == LayerRefs.TYPE_FACT:
                            reorder_result, new_layer_id = self._reorder_fact(
                                batch, l, self.network.fact_layers[l], ordinals, chain_i=i
                            )

                            if new_layer_id is not None:
                                assert reorder_result != "ignore"
                                ref_to_layer.gather = NoopGather()
                                assert new_ref_to_layer is not None

                                # Must replace references in new_ref_to_layer, because while old_ref_to_layer may have
                                # the actual ordinals (whereas new_ref_to_layer may have NoopGather), it is
                                # new_ref_to_layer that always has the reference to the fact layer.
                                self._replace_references(new_ref_to_layer, ref_from=ref, id_to=new_layer_id)
                                if self.debug:
                                    print("CHANGED", self.network.fact_layers[l])
                                if reorder_result != "found_free":
                                    i += 1
                                old_ref_to_layer = None
                        elif t == LayerRefs.TYPE_LAYER:
                            reorder_result = self._reorder_layer_output(
                                batch, layers[l], ordinals, chain_i=i, upcoming_ref=upcoming_ref
                            )
                            match reorder_result:
                                case "found" | "found_free":
                                    ref_to_layer.gather = NoopGather()
                                    if self.debug:
                                        print("CHANGED", layers[l])
                                    if reorder_result != "found_free":
                                        i += 1
                                    old_ref_to_layer = None
                                case "ignore" | "inapplicable":
                                    old_ref_to_layer = None
                                case "not_found":
                                    old_ref_to_layer = ref_to_layer
                                case _:
                                    assert False
                    case _:
                        old_ref_to_layer = ref_to_layer

    def optimize_single_use_gathers(self):
        for batch_id, batch in self.network.batches.items():
            self._for_layers(batch_id, batch.layers)


def build_optimize_single_use_gathers(max_chain_length: int | Literal["unlimited"], debug: bool):
    def optimize_single_use_gathers(network: VectorizedLayerNetwork):
        OptimizeSingleUseGathers(network, max_chain_length=max_chain_length, debug=debug).optimize_single_use_gathers()
        return network

    return optimize_single_use_gathers
