import warnings
from typing import Sequence

import numpy as np

from lib.utils import detect_repeating_K_sequence_in_list
from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation


class OptimizeKSeqRefsInLinears(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _get_refs(self, input: Input) -> Sequence | np.ndarray:
        match input:
            case GatheredLayers(gather=GenericGather(ordinals)):
                return ordinals
            case Refs():
                return np.stack([input.types, input.layer_ids, input.ordinals], axis=1)
            case _:
                assert False, f"{input}"

    def _apply_subseq_refs(self, input: Input, period: int):
        match input:
            case GatheredLayers(gather=GenericGather() as gather):
                gather.ordinals = gather.ordinals[:period]
            case Refs():
                input.types = input.types[:period]
                input.layer_ids = input.layer_ids[:period]
                input.ordinals = input.ordinals[:period]

    def _simplify(self, a: Input, b: Input, period: int):
        a_refs = self._get_refs(a)
        b_refs = self._get_refs(b)

        a_subseq = detect_repeating_K_sequence_in_list(a_refs, period=period, allow_last_incomplete=False)
        b_subseq = detect_repeating_K_sequence_in_list(b_refs, period=period, allow_last_incomplete=False)

        if a_subseq is not None and b_subseq is not None:
            warnings.warn("Linears could be simplified further")
            if len(a_refs) + len(b_subseq) < len(a_subseq) + len(b_refs):
                self._apply_subseq_refs(b, period)
            else:
                self._apply_subseq_refs(a, period)
        elif a_subseq is not None:
            self._apply_subseq_refs(a, period)
        elif b_subseq is not None:
            self._apply_subseq_refs(b, period)

    def _for_layer_base(self, base: LayerBase, period: int):
        match base:
            case InputLayerBase():
                pass
            case LinearLayerBase(input=input, weight=weight):
                self._simplify(input, weight, period)
            case LinearGatherLayerBase(input=input, weight=weight):
                self._simplify(input, weight, period)
            case _:
                assert False

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        match layer.aggregate:
            case FixedCountReduce(period=period):
                self._for_layer_base(layer.base, period)
        return layer

    def simplify_linears(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                self(bid, lid, layer)


def simplify_linears(network: VectorizedLayerNetwork):
    OptimizeKSeqRefsInLinears(network).simplify_linears()
    return network
