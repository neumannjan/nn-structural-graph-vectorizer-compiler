from collections import deque
from collections.abc import MutableMapping
from typing import Any, Iterable

from lib.vectorize.model import *


class DropUnusedLayers:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _iter_input_refs(self, input: Input) -> Iterable[tuple[int, str]]:
        match input:
            case Refs():
                for t, l, _ in input:
                    yield t, l
            case GatheredLayers(refs=refs):
                yield from refs
            case _:
                assert False, f"{input}"

    def _iter_layer_refs(self, layer: Layer) -> Iterable[tuple[int, str]]:
        match layer.base:
            case InputLayerBase(input=input):
                yield from self._iter_input_refs(input)
            case LinearLayerBase(input=input, weight=weight):
                yield from self._iter_input_refs(input)
                yield from self._iter_input_refs(weight)
            case LinearGatherLayerBase(input=input, weight=weight):
                yield from self._iter_input_refs(input)
                yield from self._iter_input_refs(weight)
            case _:
                assert False, f"{layer.base}"

    def _iter_layer_ids_from_end(self, batch: Batch) -> Iterable[str]:
        last_layer_id = next(reversed(batch.layers))

        queue = deque([last_layer_id])

        while len(queue) > 0:
            layer_id = queue.popleft()
            yield layer_id

            layer = batch.layers[layer_id]
            queue.extend(set((l for t, l in self._iter_layer_refs(layer) if t == LayerRefs.TYPE_LAYER)))

    def _find_used_facts_and_weights(self, out_fact_layers: set[str], out_weights: set[str], batch: Batch):
        for layer in batch.layers.values():
            for t, l in self._iter_layer_refs(layer):
                if t == LayerRefs.TYPE_FACT:
                    out_fact_layers.add(l)
                elif t == LayerRefs.TYPE_WEIGHT:
                    out_weights.add(l)
                elif t == LayerRefs.TYPE_LAYER:
                    pass
                else:
                    assert False, f"{t}"

    def _delete_unused_from(self, container: MutableMapping[str, Any], used: set[str]):
        unused: set[str] = set(container.keys())
        unused.difference_update(used)

        for l in unused:
            del container[l]

    def drop_unused_layers(self):
        used_fact_layers: set[str] = set()
        used_weights: set[str] = set()

        for batch in self.network.batches.values():
            used_layers = set(self._iter_layer_ids_from_end(batch))
            self._delete_unused_from(batch.layers, used_layers)

            self._find_used_facts_and_weights(used_fact_layers, used_weights, batch)

        self._delete_unused_from(self.network.fact_layers, used_fact_layers)
        self._delete_unused_from(self.network.weights, used_weights)


def drop_unused_layers(network: VectorizedLayerNetwork):
    DropUnusedLayers(network).drop_unused_layers()
    return network
