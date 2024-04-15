from collections import OrderedDict
from collections.abc import Sequence
from typing import Hashable, TypeVar, overload

from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation
from lib.vectorize.pipeline.utils.ref_groups import get_ref_groups, get_refs

_TRef = TypeVar("_TRef", bound=Hashable)


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
        return layer


class ClearOrdinalsMap(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        layer.ord_map = {}
        return layer


class OptimizeTailRefsToUniqueNoOrdRemap(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._final_layers = {batch: next(reversed(self.network.batches[batch].layers)) for batch in network.batches}

    def _remap_ref_groups_to_unique(
        self, layer: Layer, ref_groups: Sequence[tuple[_TRef, ...]]
    ) -> Sequence[tuple[_TRef, ...]]:
        # ref_groups_uniq = sorted(set(ref_groups))
        ref_groups_uniq = sorted(OrderedDict.fromkeys(ref_groups))
        group_ord_map = {group_ref: o_group_new for o_group_new, group_ref in enumerate(ref_groups_uniq)}
        layer.ord_map = {}

        for o_group, group_ref in enumerate(ref_groups):
            o_group_new = group_ord_map[group_ref]
            layer.ord_map[o_group] = o_group_new

        return ref_groups_uniq

    def _get_new_aggregate(self, aggregate: Reduce, refs_chunks_uniq: Sequence[tuple[_TRef, ...]]) -> Reduce:
        match aggregate:
            case Noop():
                return aggregate
            case FixedCountReduce():
                return aggregate
            case UnevenReduce(reduce=r):
                return UnevenReduce([len(chunk) for chunk in refs_chunks_uniq], reduce=r)
            case _:
                assert False, f"{aggregate}"

    @overload
    def _compute_ref_groups_uniq(self, layer: Layer, target: Refs) -> Sequence[tuple[tuple[int, str, int], ...]]:
        pass

    @overload
    def _compute_ref_groups_uniq(self, layer: Layer, target: GenericGather) -> Sequence[tuple[int, ...]]:
        pass

    def _compute_ref_groups_uniq(self, layer: Layer, target: Refs | GenericGather):
        refs = get_refs(target)
        ref_groups = get_ref_groups(layer.aggregate, refs)
        ref_groups_uniq = self._remap_ref_groups_to_unique(layer, ref_groups)
        return ref_groups_uniq

    def _apply_to_target(self, layer: Layer, target: Refs | GenericGather):
        match target:
            case Refs():
                ref_groups_uniq = self._compute_ref_groups_uniq(layer, target)
                refs_uniq = [o for os in ref_groups_uniq for o in os]
                target.types = [r[0] for r in refs_uniq]
                target.layer_ids = [r[1] for r in refs_uniq]
                target.ordinals = [r[2] for r in refs_uniq]
            case GenericGather():
                ref_groups_uniq = self._compute_ref_groups_uniq(layer, target)
                refs_uniq = [o for os in ref_groups_uniq for o in os]
                target.ordinals = refs_uniq
            case _:
                assert False, f"{target}"

        layer.aggregate = self._get_new_aggregate(layer.aggregate, ref_groups_uniq)

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if layer_id == self._final_layers[batch]:
            # skip the final layer
            return layer

        match layer:
            case Layer(
                base=InputLayerBase(input=Refs() as input),
            ):
                self._apply_to_target(layer, input)
            case Layer(
                base=LinearLayerBase(input=Refs() as input, weight=Refs() as weight),
                aggregate=FixedCountReduce(period=period),
            ) if period == len(weight) or period == len(input):
                if period != len(weight):
                    self._apply_to_target(layer, weight)
                elif period != len(input):
                    self._apply_to_target(layer, input)
            case Layer(
                base=LinearGatherLayerBase(input=input, weight=weight, gather=GenericGather() as gather),
            ):
                self._apply_to_target(layer, gather)

        return layer
