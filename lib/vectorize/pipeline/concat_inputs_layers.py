import numpy as np

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation


class ConcatInputsLayers(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._compute_layer_counts = ComputeLayerCounts(self.network)

    def _for_refs_with_gather(self, refs: Refs, layer_refs: LayerRefs, layer_counts: list[int]) -> GatheredLayers:
        layer_offsets: list[int] = np.concatenate([[0], np.cumsum(layer_counts[:-1], dtype=int)], dtype=int).tolist()

        offset_map: dict[tuple[int, str], int] = {
            (t, id): o for t, id, o in zip(layer_refs.types, layer_refs.layer_ids, layer_offsets)
        }

        gather = GenericGather([offset_map[t, l] + o for t, l, o in zip(refs.types, refs.layer_ids, refs.ordinals)])

        return GatheredLayers(refs=layer_refs, gather=gather)

    def _for_refs(self, batch: int, layer: str, refs: Refs) -> GatheredLayers:
        refs_uniq = sorted(set(zip(refs.types, refs.layer_ids)))
        layer_refs_uniq = LayerRefs(types=[r[0] for r in refs_uniq], layer_ids=[r[1] for r in refs_uniq])

        if (LayerRefs.TYPE_LAYER, layer) in zip(layer_refs_uniq.types, layer_refs_uniq.layer_ids):
            raise ValueError(f"Layer {layer} expects itself on input.")

        layer_counts = list(self._compute_layer_counts.iter_layer_refs_counts(batch, layer_refs_uniq))

        return self._for_refs_with_gather(refs, layer_refs_uniq, layer_counts)

    def _for_input(self, batch: int, layer: str, input: Input):
        match input:
            case Refs():
                return self._for_refs(batch, layer, input)
            case GatheredLayers():
                return input
            case _:
                assert False, f"{input}"

    def _for_layer_base(self, batch: int, layer: str, base: LayerBase):
        match base:
            case InputLayerBase(input=input):
                input = self._for_input(batch, layer, input)
                return InputLayerBase(input=input)
            case LinearLayerBase(input=input, weight=weight, lifts=lifts):
                input = self._for_input(batch, layer, input)
                weight = self._for_input(batch, layer, weight)
                return LinearLayerBase(input=input, weight=weight, lifts=lifts)
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather, lifts=lifts):
                input = self._for_input(batch, layer, input)
                weight = self._for_input(batch, layer, weight)
                return LinearGatherLayerBase(input=input, weight=weight, gather=gather, lifts=lifts)
            case _:
                assert False

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        layer.base = self._for_layer_base(batch, layer_id, layer.base)
        return layer

    def concat_inputs_layers(self):
        for bid, batch in self.network.batches.items():
            for layer_id, layer in batch.layers.items():
                layer.base = self._for_layer_base(bid, layer_id, layer.base)


def concat_inputs_layers(network: VectorizedLayerNetwork):
    ConcatInputsLayers(network).concat_inputs_layers()
    return network
