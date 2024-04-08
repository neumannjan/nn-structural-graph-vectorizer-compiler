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


class ConvertRefsToUniqueNoOrdRemap(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._final_layers = {batch: next(reversed(self.network.batches[batch].layers)) for batch in network.batches}

    def _rid_tail_ordinals(self, layer: Layer, ordinals: list[int]):
        layer.ord_map = {o: o_real for o, o_real in enumerate(ordinals) if o != o_real}

    def _rid_tail_refs(self, layer: Layer, refs: Refs) -> Refs:
        refs_uniq = sorted(set(refs))
        refs_ord_map = {ref: o_real for o_real, ref in enumerate(refs_uniq)}
        layer.ord_map = {}

        for o, ref in enumerate(refs):
            o_real = refs_ord_map[ref]
            if o_real != o:
                layer.ord_map[o] = o_real

        return Refs(
            types=[r[0] for r in refs_uniq],
            layer_ids=[r[1] for r in refs_uniq],
            ordinals=[r[2] for r in refs_uniq],
        )

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if layer_id == self._final_layers[batch]:
            # skip the final layer
            return layer

        # # TODO: find a solution for when aggregate has a value
        match layer:
            case Layer(
                base=InputLayerBase(input=Refs() as input) as base,
                aggregate=Noop(),
            ):
                base.input = self._rid_tail_refs(layer, input)
                return layer
            case Layer(
                base=LinearGatherLayerBase(input=input, weight=weight, gather=GenericGather(ordinals)),
                aggregate=Noop(),
            ):
                self._rid_tail_ordinals(layer, ordinals)
                layer.base = LinearLayerBase(input=input, weight=weight)
                return layer
            case _:
                return layer


def convert_refs_to_unique(network: VectorizedLayerNetwork):
    return LayerwiseSeq(
        RemapOrdinals,
        ConvertRefsToUniqueNoOrdRemap,
    )
