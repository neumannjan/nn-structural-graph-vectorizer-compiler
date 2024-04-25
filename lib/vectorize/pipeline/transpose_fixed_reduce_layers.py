from typing import Sequence, TypeVar

from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation

_T = TypeVar("_T")


class TransposeFixedCountReduceLayers(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        pass

    def _transpose_seq(self, seq: Sequence[_T], period: int) -> list[_T]:
        out: list[_T] = []

        for i in range(period):
            out.extend(seq[i::period])

        return out

    def _transpose_input(self, input: Input, period: int) -> None:
        match input:
            case Refs():
                input.types = self._transpose_seq(input.types, period)
                input.layer_ids = self._transpose_seq(input.layer_ids, period)
                input.ordinals = self._transpose_seq(input.ordinals, period)
            case _:
                raise NotImplementedError()

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        match layer:
            case Layer(
                base=LinearLayerBase(input=input, weight=weight, lifts=None),
                aggregate=FixedCountReduce(period=period, dim=1) as aggregate,
            ):
                self._transpose_input(input, period)
                self._transpose_input(weight, period)
                aggregate.dim = 0
            case Layer(base=LinearLayerBase()) | Layer(base=InputLayerBase()):
                pass
            case _:
                raise ValueError(layer)

        return layer
