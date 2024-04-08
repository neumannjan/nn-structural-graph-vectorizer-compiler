from collections.abc import Iterable
from typing import Collection, Hashable, TypeVar

from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation, LayerwiseSeq


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


_TRef = TypeVar("_TRef", bound=Hashable)


def _grouper(iterable: Iterable[_TRef], n: int, fillvalue=None) -> Iterable[tuple[_TRef, ...]]:
    it = iter(iterable)
    while True:
        try:
            yield tuple([next(it) for _ in range(n)])
        except StopIteration:
            break


def _uneven_grouper(iterable: Iterable[_TRef], counts: list[int]) -> Iterable[tuple[_TRef, ...]]:
    it = iter(iterable)

    for c in counts:
        yield tuple((next(it) for _ in range(c)))


class ConvertRefsToUniqueNoOrdRemap(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._final_layers = {batch: next(reversed(self.network.batches[batch].layers)) for batch in network.batches}

    def _remap_tail_refs_to_unique(
        self, layer: Layer, refs_chunks: Collection[tuple[_TRef, ...]]
    ) -> list[tuple[_TRef, ...]]:
        refs_chunks_uniq = sorted(set(refs_chunks))
        ord_chunk_map = {chunk_ref: o_chunk_real for o_chunk_real, chunk_ref in enumerate(refs_chunks_uniq)}
        layer.ord_map = {}

        for o_chunk, chunk_ref in enumerate(refs_chunks):
            o_chunk_real = ord_chunk_map[chunk_ref]
            layer.ord_map[o_chunk] = o_chunk_real

        return refs_chunks_uniq

    def _get_ref_chunks(self, aggregate: Reduce, refs: Collection[_TRef]) -> Collection[tuple[_TRef, ...]]:
        match aggregate:
            case Noop():
                return [(r,) for r in refs]
            case FixedCountReduce(period=period):
                return list(_grouper(refs, n=period))
            case UnevenReduce(counts=counts):
                return list(_uneven_grouper(refs, counts=counts))
            case _:
                assert False, f"{aggregate}"

    def _get_new_aggregate(self, aggregate: Reduce, refs_chunks_uniq: Collection[tuple[_TRef, ...]]) -> Reduce:
        match aggregate:
            case Noop():
                return aggregate
            case FixedCountReduce():
                return aggregate
            case UnevenReduce(reduce=r):
                return UnevenReduce([len(chunk) for chunk in refs_chunks_uniq], reduce=r)
            case _:
                assert False, f"{aggregate}"

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if layer_id == self._final_layers[batch]:
            # skip the final layer
            return layer

        # # TODO: find a solution for when aggregate has a value
        match layer:
            case Layer(
                base=InputLayerBase(input=Refs() as input) as base,
                aggregate=aggregate,
            ):
                ref_chunks = self._get_ref_chunks(aggregate, input)

                refs_chunks_uniq = self._remap_tail_refs_to_unique(layer, ref_chunks)
                refs_uniq = [o for os in refs_chunks_uniq for o in os]

                base.input = Refs(
                    types=[r[0] for r in refs_uniq],
                    layer_ids=[r[1] for r in refs_uniq],
                    ordinals=[r[2] for r in refs_uniq],
                )

                layer.aggregate = self._get_new_aggregate(aggregate, refs_chunks_uniq)
                return layer
            case Layer(
                base=LinearGatherLayerBase(input=input, weight=weight, gather=GenericGather(ordinals) as gather),
                aggregate=aggregate,
            ):
                ref_chunks = self._get_ref_chunks(aggregate, ordinals)

                refs_chunks_uniq = self._remap_tail_refs_to_unique(layer, ref_chunks)
                refs_uniq = [o for os in refs_chunks_uniq for o in os]

                gather.ordinals = refs_uniq

                layer.aggregate = self._get_new_aggregate(aggregate, refs_chunks_uniq)
                return layer
            case _:
                return layer


def convert_refs_to_unique(network: VectorizedLayerNetwork):
    return LayerwiseSeq(
        RemapOrdinals,
        ConvertRefsToUniqueNoOrdRemap,
    )
