from typing import Iterable, Iterator, OrderedDict

import numpy as np

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from compute_graph_vectorize.vectorize.pipeline.compute_layer_shapes import assert_shapes_equal, reduce_shapes

_Grp = tuple[int, tuple[str, ...]]


def _iter_grp_refs(grp: _Grp):
    t, ids = grp
    for id in ids:
        yield t, id


class MergeTrivialLayerConcats:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._counts = ComputeLayerCounts(network)

    def _get_ref_groups(self, refs: LayerRefs) -> Iterable[_Grp]:
        the_type: int | None = None
        out: list[str] = []

        for t, l in refs:
            if t == the_type:
                out.append(l)
            else:
                if the_type is not None and out:
                    yield the_type, tuple(out)

                if t not in (LayerRefs.TYPE_FACT, LayerRefs.TYPE_WEIGHT):
                    the_type = None
                    continue

                the_type = t
                out = [l]

        if the_type is not None and out:
            yield the_type, tuple(out)

    def _get_ref_groups_in_input(self, input: Input):
        match input:
            case GatheredLayers(refs=refs):
                return self._get_ref_groups(refs)
            case _:
                assert False, f"{input}"

    def _get_ref_groups_in_layer(self, layer: Layer):
        match layer.base:
            case InputLayerBase(input=input):
                yield from self._get_ref_groups_in_input(input)
            case LinearLayerBase(input=input, weight=weight):
                yield from self._get_ref_groups_in_input(input)
                yield from self._get_ref_groups_in_input(weight)
            case LinearGatherLayerBase(input=input, weight=weight):
                yield from self._get_ref_groups_in_input(input)
                yield from self._get_ref_groups_in_input(weight)
            case _:
                assert False, f"{layer.base}"

    def _find_unique_ref_groups(self):
        blacklist: set[tuple[int, str]] = set()
        out: dict[tuple[int, str], _Grp] = {}

        def _has_overlap(grp: _Grp):
            weight_set = set(w for t, w in _iter_grp_refs(grp) if t == Refs.TYPE_WEIGHT)

            for ref in _iter_grp_refs(grp):
                if ref in blacklist or out.get(ref, None) not in (None, grp):
                    return True

                t, w = ref
                if t == Refs.TYPE_WEIGHT:
                    if w in weight_set:
                        weight_set.remove(w)
                    else:
                        return True

            return False

        def _blacklist_grp(grp: _Grp):
            for ref in _iter_grp_refs(grp):
                blacklist.add(ref)
                out.pop(ref, None)

        def _apply_grp(grp: _Grp):
            for ref in _iter_grp_refs(grp):
                out[ref] = grp

        for batch in self.network.batches.values():
            for layer in batch.layers.values():
                for grp in self._get_ref_groups_in_layer(layer):
                    if _has_overlap(grp) or len(grp[1]) <= 1:
                        _blacklist_grp(grp)
                    else:
                        _apply_grp(grp)

        return out

    def _get_new_group_id(self, ids: Iterable[str]) -> str:
        return "_".join(OrderedDict.fromkeys(ids))

    def _get_new_group_ref(self, grp: _Grp) -> tuple[int, str]:
        t, ids = grp
        return t, self._get_new_group_id(ids)

    def _create_merged_values(self, ref_groups: Iterable[_Grp]):
        for t, ids in ref_groups:
            grp_id = self._get_new_group_id(ids)
            if t == LayerRefs.TYPE_FACT:
                all_facts = [self.network.fact_layers[id] for id in ids]
                facts = [f for facts_this in all_facts for f in facts_this.facts]
                self.network.fact_layers[grp_id] = FactLayer(
                    facts=facts,
                    count=self._counts.compute_facts_count(facts),
                    shape=reduce_shapes((f.shape for f in all_facts), assert_shapes_equal),
                )
            elif t == LayerRefs.TYPE_WEIGHT:
                all_weights = [self.network.weights[id].value for id in ids]
                self.network.weights[grp_id] = LearnableWeight(np.concatenate(all_weights, axis=0))
            else:
                assert False, f"{t}"

    def _iter_refs_groupwise(
        self, refs: LayerRefs, ref_groups: dict[tuple[int, str], _Grp]
    ) -> Iterator[tuple[int, str]]:
        it = iter(refs)

        the_grp: _Grp | None = None
        the_grp_it: Iterator[tuple[int, str]] | None = None
        try:
            while True:
                head = next(it)

                this_grp = ref_groups.get(head, None)

                # previous group is finished if there is any and there is no match
                if the_grp is not None and this_grp != the_grp:
                    assert the_grp_it is not None
                    # assert it is actually finished
                    try:
                        next(the_grp_it)
                        assert False
                    except StopIteration:
                        pass

                    yield self._get_new_group_ref(the_grp)
                    the_grp = None

                # if not group for this node, yield it immediately
                if this_grp is None:
                    yield head
                    continue

                # if just starting a new group, build the iterator
                if the_grp is None and this_grp is not None:
                    the_grp = this_grp
                    the_grp_it = iter(_iter_grp_refs(the_grp))

                # assert the matching value
                assert the_grp_it is not None
                try:
                    head2 = next(the_grp_it)
                    assert head == head2
                except StopIteration:
                    raise ValueError()
        except StopIteration:
            pass

        # final group is finished, if there is any
        if the_grp is not None:
            assert the_grp_it is not None
            # assert it is actually finished
            try:
                next(the_grp_it)
                assert False
            except StopIteration:
                pass

            yield self._get_new_group_ref(the_grp)

    def _apply_merged_refs(self, batch: int, ref_groups: dict[tuple[int, str], _Grp], refs: LayerRefs):
        out_types: list[int] = []
        out_layer_ids: list[str] = []
        for t, l in self._iter_refs_groupwise(refs, ref_groups):
            out_types.append(t)
            out_layer_ids.append(l)

        refs.types = out_types
        refs.layer_ids = out_layer_ids

    def _apply_merged_refs_in_input(self, batch: int, ref_groups: dict[tuple[int, str], _Grp], input: Input):
        match input:
            case GatheredLayers(refs=refs):
                self._apply_merged_refs(batch, ref_groups, refs)
            case _:
                assert False, f"{input}"

    def _apply_merged_refs_in_layer(self, batch: int, ref_groups: dict[tuple[int, str], _Grp], layer: Layer):
        match layer.base:
            case InputLayerBase(input=input):
                self._apply_merged_refs_in_input(batch, ref_groups, input)
            case LinearLayerBase(input=input, weight=weight):
                self._apply_merged_refs_in_input(batch, ref_groups, input)
                self._apply_merged_refs_in_input(batch, ref_groups, weight)
            case LinearGatherLayerBase(input=input, weight=weight):
                self._apply_merged_refs_in_input(batch, ref_groups, input)
                self._apply_merged_refs_in_input(batch, ref_groups, weight)
            case _:
                assert False, f"{layer.base}"

    def _apply_merged_refs_in_network(self, ref_groups: dict[tuple[int, str], _Grp]):
        for batch_id, batch in self.network.batches.items():
            for layer in batch.layers.values():
                self._apply_merged_refs_in_layer(batch_id, ref_groups, layer)

    def merge_trivial_layer_concats(self):
        ref_groups = self._find_unique_ref_groups()
        self._create_merged_values(set(ref_groups.values()))
        self._apply_merged_refs_in_network(ref_groups)


def merge_trivial_layer_concats(network: VectorizedLayerNetwork):
    MergeTrivialLayerConcats(network).merge_trivial_layer_concats()
    return network
