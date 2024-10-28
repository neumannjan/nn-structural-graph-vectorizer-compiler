from collections import OrderedDict

from compute_graph_vectorize.vectorize.model import *


class GiveUniqueNames:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _for_refs(self, refs: Refs | LayerRefs):
        for i, t in enumerate(refs.types):
            if t == Refs.TYPE_FACT:
                prefix = "f_"
            elif t == Refs.TYPE_WEIGHT:
                prefix = "w_"
            elif t == Refs.TYPE_LAYER:
                prefix = "l_"
            else:
                assert False, t

            refs.layer_ids[i] = prefix + refs.layer_ids[i]

    def _for_input(self, input: Input):
        match input:
            case Refs():
                self._for_refs(input)
            case GatheredLayers(refs=refs):
                self._for_refs(refs)
            case _:
                assert False, f"{input}"

    def _for_layer(self, layer: Layer) -> Layer:
        match layer.base:
            case InputLayerBase(input=input):
                self._for_input(input)
            case LinearLayerBase(input=input, weight=weight):
                self._for_input(input)
                self._for_input(weight)
            case LinearGatherLayerBase(input=input, weight=weight):
                self._for_input(input)
                self._for_input(weight)
            case _:
                assert False

        return layer

    def give_unique_names(self):
        self.network.fact_layers = {"f_" + k: v for k, v in self.network.fact_layers.items()}
        self.network.weights = {"w_" + k: v for k, v in self.network.weights.items()}

        for batch in self.network.batches.values():
            batch.layers = OrderedDict((("l_" + k, self._for_layer(v)) for k, v in batch.layers.items()))


def give_unique_names(network: VectorizedLayerNetwork):
    GiveUniqueNames(network).give_unique_names()
    return network
