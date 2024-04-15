import warnings
from typing import Sequence

import numpy as np

from lib.utils import detect_repeating_K_sequence_in_list
from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation


class OptimizeSequenceRefsInLinears(LayerwiseOperation):
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

    def _simplify(self, a: Input, b: Input, preferred_period: int | None):
        a_refs = self._get_refs(a)
        b_refs = self._get_refs(b)

        if preferred_period is None:
            # TODO
            return

        a_subseq_len = detect_repeating_K_sequence_in_list(a_refs, period=preferred_period, allow_last_incomplete=False)
        b_subseq_len = detect_repeating_K_sequence_in_list(b_refs, period=preferred_period, allow_last_incomplete=False)

        if a_subseq_len is not None and b_subseq_len is not None:
            warnings.warn("Linears could be simplified further")
            if len(a_refs) + b_subseq_len < a_subseq_len + len(b_refs):
                self._apply_subseq_refs(b, preferred_period)
            else:
                self._apply_subseq_refs(a, preferred_period)
        elif a_subseq_len is not None:
            self._apply_subseq_refs(a, preferred_period)
        elif b_subseq_len is not None:
            self._apply_subseq_refs(b, preferred_period)

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
        match layer:
            case Layer(base=InputLayerBase()):
                pass
            case Layer(
                base=(LinearLayerBase(input=input, weight=weight) | LinearGatherLayerBase(input=input, weight=weight)),
                aggregate=FixedCountReduce(period=period),
            ):
                self._simplify(input, weight, preferred_period=period)
            case Layer(
                base=(LinearLayerBase(input=input, weight=weight) | LinearGatherLayerBase(input=input, weight=weight))
            ):
                self._simplify(input, weight, preferred_period=None)
            case _:
                assert False, f"{layer}"
        return layer

    def optimize_sequence_refs_in_linears(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                self(bid, lid, layer)


def optimize_sequence_refs_in_linears(network: VectorizedLayerNetwork):
    OptimizeSequenceRefsInLinears(network).optimize_sequence_refs_in_linears()
    return network
