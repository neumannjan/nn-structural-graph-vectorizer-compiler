from collections.abc import Sequence
from typing import Iterable, Iterator, Literal, TypeVar

from lib.vectorize.model import *
from lib.vectorize.model.layer import get_lifts_period
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.utils.chain_graph import ComputeChainGraph
from lib.vectorize.pipeline.utils.gather import combine_gathers
from lib.vectorize.pipeline.utils.ref_groups import build_grouper_for_aggregate


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


def _get_aggregate_period(aggregate: Reduce) -> int | None:
    match aggregate:
        case Noop():
            return 1
        case FixedCountReduce(period=period):
            return period
        case UnevenReduce():
            return None
        case _:
            raise ValueError(aggregate)


class OptimizeSingleUseGathers:
    def __init__(
        self,
        network: VectorizedLayerNetwork,
        max_chain_length: int | Literal["unlimited"],
        propagate_through_symmetries: bool,
        debug: bool,
    ) -> None:
        self.network = network
        self.max_chain_length = float("inf") if max_chain_length == "unlimited" else max_chain_length
        self.propagate_through_symmetries = propagate_through_symmetries
        self.debug = debug

        self._compute_chain_graph = ComputeChainGraph(network, types=(LayerRefs.TYPE_LAYER, LayerRefs.TYPE_FACT))
        self._counts = ComputeLayerCounts(network)

        self._fact_reorders: dict[tuple[str, tuple[int, ...]], str] = {}

    def _reorder_fact(
        self, batch_id: int, orig_layer_id: str, fact_layer: FactLayer, ordinals2: list[int], chain_i: int
    ) -> tuple[Literal["found", "found_free"], str] | tuple[Literal["ignore"], None]:
        reordered_key = orig_layer_id, tuple(ordinals2)

        new_layer_id = self._fact_reorders.get(reordered_key, None)

        # We must create this version of the layer first.
        if new_layer_id is None:
            facts_new = [fact_layer.facts[o] for o in ordinals2]
            count_new = self._counts.compute_facts_count(facts_new)
            new_layer = FactLayer(facts=facts_new, count=count_new, shape=fact_layer.shape)
        else:
            new_layer = self.network.fact_layers[new_layer_id]
            count_new = new_layer.count
            assert count_new is not None

        count_old = self._counts.compute_facts_count(fact_layer.facts) if fact_layer.count is None else fact_layer.count

        is_free = count_new <= count_old

        # Let's not do it and use the original.
        if not is_free and chain_i >= self.max_chain_length:
            return "ignore", None

        if new_layer_id is None:
            new_layer_id = orig_layer_id + "__" + str(batch_id)

            self.network.fact_layers[new_layer_id] = new_layer
            self._fact_reorders[reordered_key] = new_layer_id

        return ("found_free", new_layer_id) if is_free else ("found", new_layer_id)

    def _get_gather_ordinals(
        self, batch: int, refs: LayerRefs | tuple[Input, Input], gather: GenericGather | NoopGather
    ) -> Sequence[int]:
        match gather:
            case GenericGather(ordinals):
                return ordinals
            case NoopGather():
                match refs:
                    case LayerRefs():
                        in_count = self._counts.compute_layer_refs_count(batch, refs)
                    case (input, weight):
                        in_count = self._counts.compute_linear_count(batch, input, weight, None)
                    case _:
                        assert False, f"{refs}"

                return range(in_count)
            case _:
                assert False, f"{gather}"

    def _compute_refs_count(self, batch: int, refs: LayerRefs | tuple[Input, Input]):
        match refs:
            case LayerRefs():
                return self._counts.compute_layer_refs_count(batch, refs)
            case (input, weight):
                return self._counts.compute_linear_count(batch, input, weight, None)
            case _:
                raise ValueError(refs)

    def _do_reorder_layer_output(
        self,
        batch: int,
        gather: GenericGather | NoopGather,
        refs: LayerRefs | tuple[Input, Input],
        container: GatheredLayers,
        other: Iterable[
            tuple[
                GenericGather | NoopGather,
                LayerRefs | tuple[Input, Input],
                GatheredLayers,
            ]
        ],
        aggregate: Reduce,
        ordinals2: list[int],
        chain_i: int,
    ) -> Literal["ignore", "found_free", "found"]:
        grouper = build_grouper_for_aggregate(aggregate)
        first_ord_groups = dict(enumerate(grouper.group(self._get_gather_ordinals(batch, refs, gather))))
        gather_new_groups = [first_ord_groups[o] for o in ordinals2]
        gather_new = GenericGather(list(grouper.ungroup(gather_new_groups)))

        refs_cnt = self._compute_refs_count(batch, refs)
        gather_cnt = self._counts.compute_gather_count(refs_cnt, gather)
        gather_new_cnt = self._counts.compute_gather_count(refs_cnt, gather_new)

        is_free = gather_new_cnt <= gather_cnt

        if not is_free and chain_i >= self.max_chain_length:
            return "ignore"

        container.gather = gather_new

        for gather_this, refs_this, container_this in other:
            first_ord_groups = dict(enumerate(grouper.group(self._get_gather_ordinals(batch, refs_this, gather_this))))
            gather_this_new_groups = [first_ord_groups[o] for o in ordinals2]
            gather_this_new = GenericGather(list(grouper.ungroup(gather_this_new_groups)))
            container_this.gather = gather_this_new

        _reorder_aggregate(aggregate, ordinals2)
        return "found_free" if is_free else "found"

    def _reorder_layer_output(
        self, batch: int, layer: Layer, ordinals2: list[int], chain_i: int, upcoming_ref: tuple[int, str] | None
    ) -> Literal["ignore", "found_free", "found", "propagate", "inapplicable"]:
        if layer.compilable:
            return "inapplicable"

        match layer:
            case Layer(base=LinearGatherLayerBase()):
                raise ValueError(layer.base)
            case Layer(base=InputLayerBase(input=GatheredLayers(refs=[_], gather=NoopGather()), aggregate=Noop())):
                return "propagate"
            case Layer(
                base=(
                    LinearLayerBase(input=GatheredLayers(refs=[ref2], gather=NoopGather()) as this, lifts=None)
                ) as base,
                aggregate=Noop(),
            ) if upcoming_ref == ref2:
                return "propagate"
            case Layer(
                base=LinearLayerBase(
                    input=GatheredLayers(refs=refs_a, gather=gather_a) as input,
                    weight=GatheredLayers(refs=refs_b, gather=gather_b) as weight,
                    lifts=lifts,
                ) as base,
                aggregate=aggregate,
            ) if self.propagate_through_symmetries and lifts is not None and _get_aggregate_period(
                aggregate
            ) == get_lifts_period(lifts):
                assert isinstance(gather_a, (GenericGather, NoopGather))
                assert isinstance(gather_b, (GenericGather, NoopGather))

                input_count = self._counts.compute_input_count(batch, input)
                weight_count = self._counts.compute_input_count(batch, weight)

                layer_count = self._counts.compute_linear_count(batch, input, weight, lifts)

                variants: list[
                    tuple[
                        GenericGather | NoopGather,
                        LayerRefs | tuple[Input, Input],
                        GatheredLayers,
                    ]
                ] = []

                if input_count == layer_count:
                    variants.append((gather_a, refs_a, input))
                if weight_count == layer_count:
                    variants.append((gather_b, refs_b, weight))

                if len(variants) == 0:
                    return "inapplicable"

                return self._do_reorder_layer_output(
                    batch,
                    *variants[0],
                    variants[1:],
                    aggregate,
                    ordinals2,
                    chain_i,
                )
            case Layer(
                base=(
                    LinearLayerBase(
                        weight=GatheredLayers(refs=[ref2], gather=NoopGather()) as this,
                        lifts=None,
                    )
                ) as base,
                aggregate=Noop(),
            ) if upcoming_ref == ref2:
                return "propagate"
            case Layer(
                base=LinearLayerBase(
                    input=GatheredLayers(refs=refs_a, gather=gather_a) as input,
                    weight=GatheredLayers(refs=refs_b, gather=gather_b) as weight,
                    lifts=None,
                ),
                aggregate=aggregate,
            ):
                assert isinstance(gather_a, (GenericGather, NoopGather))
                assert isinstance(gather_b, (GenericGather, NoopGather))

                input_count = self._counts.compute_input_count(batch, input)
                weight_count = self._counts.compute_input_count(batch, weight)

                assert input_count == weight_count

                return self._do_reorder_layer_output(
                    batch,
                    gather_a,
                    refs_a,
                    input,
                    ((gather_b, refs_b, weight),),
                    aggregate,
                    ordinals2,
                    chain_i,
                )
            case Layer(
                base=InputLayerBase(input=GatheredLayers(refs=refs, gather=gather) as this) as base,
                aggregate=aggregate,
            ):
                assert isinstance(gather, (GenericGather, NoopGather))
                return self._do_reorder_layer_output(
                    batch,
                    gather,
                    refs,
                    this,
                    (),
                    aggregate,
                    ordinals2,
                    chain_i,
                )
            case Layer(base=LinearLayerBase(lifts=lifts)) if lifts is not None:
                return "inapplicable"

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
            if self.debug:
                print(chain)
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
                            else:
                                i = 0
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
                                    if reorder_result == "ignore":
                                        print(f"IGNORING (chain_i={i}, max_chain_length={self.max_chain_length})")
                                    i = 0
                                    old_ref_to_layer = None
                                case "propagate":
                                    old_ref_to_layer = ref_to_layer
                                case _:
                                    assert False
                    case _:
                        old_ref_to_layer = ref_to_layer

    def optimize_single_use_gathers(self):
        for batch_id, batch in self.network.batches.items():
            self._for_layers(batch_id, batch.layers)


def build_optimize_single_use_gathers(
    max_chain_length: int | Literal["unlimited"], propagate_through_symmetries: bool, debug: bool
):
    def optimize_single_use_gathers(network: VectorizedLayerNetwork):
        OptimizeSingleUseGathers(
            network,
            max_chain_length=max_chain_length,
            propagate_through_symmetries=propagate_through_symmetries,
            debug=debug,
        ).optimize_single_use_gathers()
        return network

    return optimize_single_use_gathers
