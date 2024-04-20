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
        self._op_factories = list(op_factories)

    def __add__(self, other: "Layerwise | _LayerwiseOpFactory | LayerwiseOperation") -> "Layerwise":
        if isinstance(other, Layerwise):
            return Layerwise(*self._op_factories, *other._op_factories)
        else:
            return Layerwise(*self._op_factories, other)

    def __iadd__(self, other: "Layerwise | _LayerwiseOpFactory | LayerwiseOperation") -> "Layerwise":
        if isinstance(other, Layerwise):
            self._op_factories.extend(other._op_factories)
        else:
            self._op_factories.append(other)

        return self

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

    def __len__(self) -> int:
        return len(self._op_factories)


class LayerwisePrint(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        pass

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        print(layer_id)
        print(layer)
        return layer
