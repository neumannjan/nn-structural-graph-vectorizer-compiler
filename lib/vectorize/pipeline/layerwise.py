from collections.abc import Sequence
from typing import Protocol

from lib.vectorize.model import *


class LayerwiseOperation(Protocol):
    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer: ...


class _LayerwiseOpFactory(Protocol):
    def __call__(self, network: VectorizedLayerNetwork) -> LayerwiseOperation: ...


class Layerwise:
    def __init__(self, op_factory: _LayerwiseOpFactory) -> None:
        self._op_factory = op_factory

    def __call__(self, network: VectorizedLayerNetwork) -> VectorizedLayerNetwork:
        op = self._op_factory(network)

        for bid, batch in network.batches.items():
            for lid, layer in batch.layers.items():
                try:
                    out_layer = op(bid, lid, layer)

                    if out_layer != layer:
                        batch.layers[lid] = out_layer
                except Exception as e:
                    raise Exception(f"Exception in layer {lid} (batch {bid})") from e

        return network


class _LayerwiseOpSeqFactory(_LayerwiseOpFactory):
    def __init__(self, ops: Sequence[_LayerwiseOpFactory]) -> None:
        self._op_factories = ops

    def __call__(self, network: VectorizedLayerNetwork) -> LayerwiseOperation:
        ops = [f(network) for f in self._op_factories]
        return _LayerwiseOpSeq(ops)


class _LayerwiseOpSeq(LayerwiseOperation):
    def __init__(self, ops: Sequence[LayerwiseOperation]) -> None:
        self._ops = ops

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        for op in self._ops:
            layer = op(batch, layer_id, layer)

        return layer


class LayerwiseSeq(Layerwise):
    def __init__(self, *op_factories: _LayerwiseOpFactory) -> None:
        super().__init__(_LayerwiseOpSeqFactory(op_factories))
