from torch_geometric.utils.sparse import warnings

from lib.utils import detect_repeating_K_sequence_in_list
from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation


class SimplifyLinears(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _simplify(self, a: GenericGather, b: GenericGather, period: int) -> tuple[Gather, Gather]:
        a_subseq = detect_repeating_K_sequence_in_list(a.ordinals, period=period, allow_last_incomplete=False)
        b_subseq = detect_repeating_K_sequence_in_list(b.ordinals, period=period, allow_last_incomplete=False)

        if a_subseq is not None and b_subseq is not None:
            warnings.warn("Linears could be simplified further")
            return GenericGather(a_subseq.tolist()), b
        elif a_subseq is not None:
            return GenericGather(a_subseq.tolist()), b
        elif b_subseq is not None:
            return a, GenericGather(b_subseq.tolist())
        else:
            return a, b

    def _for_layer_base(self, base: LayerBase, period: int) -> LayerBase:
        match base:
            case InputLayerBase():
                return base
            case LinearLayerBase(
                input=GatheredLayers(gather=GenericGather() as a) as input,
                weight=GatheredLayers(gather=GenericGather() as b) as weight,
            ):
                input.gather, weight.gather = self._simplify(a, b, period)
                return base
            case LinearGatherLayerBase(
                input=GatheredLayers(gather=GenericGather() as a) as input,
                weight=GatheredLayers(gather=GenericGather() as b) as weight,
            ):
                input.gather, weight.gather = self._simplify(a, b, period)
                return base
            case _:
                assert False

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        match layer.aggregate:
            case FixedCountReduce(period=period):
                layer.base = self._for_layer_base(layer.base, period)
        return layer

    def simplify_linears(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                self(bid, lid, layer)


def simplify_linears(network: VectorizedLayerNetwork):
    SimplifyLinears(network).simplify_linears()
    return network
