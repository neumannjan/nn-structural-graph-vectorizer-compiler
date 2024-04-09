from abc import ABC, abstractmethod
from typing import Protocol

from lib.vectorize.model import *


class LayerwiseOperation(ABC):
    @abstractmethod
    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer: ...


class _LayerwiseOpFactory(Protocol):
    def __call__(self, network: VectorizedLayerNetwork) -> LayerwiseOperation: ...


class Layerwise:
    def __init__(self, *op_factories: _LayerwiseOpFactory | LayerwiseOperation) -> None:
        self._op_factories = op_factories

    def __call__(self, network: VectorizedLayerNetwork) -> VectorizedLayerNetwork:
        ops: list[LayerwiseOperation] = [
            f if isinstance(f, LayerwiseOperation) else f(network) for f in self._op_factories
        ]

        for bid, batch in network.batches.items():
            for lid, layer in batch.layers.items():
                try:
                    out_layer = layer

                    for op in ops:
                        out_layer = op(bid, lid, out_layer)

                    if out_layer != layer:
                        batch.layers[lid] = out_layer
                except Exception as e:
                    raise Exception(f"Exception in layer {lid} (batch {bid})") from e

        return network


class LayerwisePrint(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        pass

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        print(layer)
        return layer
