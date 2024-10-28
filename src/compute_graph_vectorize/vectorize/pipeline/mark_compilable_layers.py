from typing import Iterator

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.layerwise import LayerwiseOperation


class MarkCompilableLayers(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _iter_all_layer_refs(self, layer: Layer) -> Iterator[tuple[int, str]]:
        match layer.base:
            case InputLayerBase(input=GatheredLayers() as input):
                yield from input.refs
            case (
                LinearLayerBase(input=GatheredLayers() as input, weight=GatheredLayers() as weight)
                | LinearGatherLayerBase(input=GatheredLayers() as input, weight=GatheredLayers() as weight)
            ):
                yield from input.refs
                yield from weight.refs

    def _is_layer_ref_compilable(self, batch: int, ref: tuple[int, str]) -> bool:
        t, l = ref

        match t:
            case LayerRefs.TYPE_WEIGHT:
                return False
            case LayerRefs.TYPE_FACT:
                return True
            case LayerRefs.TYPE_LAYER:
                return self.network.batches[batch].layers[l].compilable
            case _:
                raise ValueError(t)

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if layer.compilable:
            return layer

        if all((self._is_layer_ref_compilable(batch, ref) for ref in self._iter_all_layer_refs(layer))):
            layer.compilable = True
        return layer
