import numpy as np

from lib.vectorize.model import *


class ConcatInputsLayers:
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network
        self.layer_ref_pool = LayerRefPool()

    def _map_ref_to_layer_ref(self, ref: Ref) -> LayerRef:
        match ref:
            case FactRef(id=id, ordinal=_):
                return self.layer_ref_pool.fact(id)
            case NeuronRef(id=id, ordinal=_):
                return self.layer_ref_pool.neuron(id=id)
            case WeightRef():
                return ref
            case _:
                assert False, f"{ref}"

    def _get_count(self, batch: int, layer_ref: LayerRef) -> int:
        match layer_ref:
            case FactLayerRef(id=id):
                cnt = self.network.fact_layers[id].count
                assert cnt is not None
                return cnt
            case NeuronLayerRef(id=id):
                cnt = self.network.batches[batch].layers[id].count
                assert cnt is not None
                return cnt
            case WeightRef(id=id):
                return self.network.weights[id].value.shape[0]
            case _:
                assert False, f"{layer_ref}"

    def _get_ref_offset(self, ref: Ref) -> int:
        match ref:
            case FactRef(ordinal=o):
                return o
            case NeuronRef(ordinal=o):
                return o
            case WeightRef():
                return 0
            case _:
                assert False

    def _for_refs(self, batch: int, refs: Refs) -> GatheredLayers:
        layer_refs = [self._map_ref_to_layer_ref(ref) for ref in refs.refs]
        layer_counts: dict[LayerRef, int] = dict()

        for ref in layer_refs:
            if ref not in layer_counts:
                layer_counts[ref] = self._get_count(batch, ref)

        layer_refs_uniq = sorted(layer_counts.keys(), reverse=False)
        layer_sizes_list = np.array([layer_counts[l] for l in layer_refs_uniq[:-1]])
        layer_offsets: list[int] = np.concatenate([[0], np.cumsum(layer_sizes_list)]).tolist()
        layer_to_offset_map: dict[LayerRef, int] = {id: off for id, off in zip(layer_refs_uniq, layer_offsets)}

        gather = GenericGather(
            [layer_to_offset_map[self._map_ref_to_layer_ref(ref)] + self._get_ref_offset(ref) for ref in refs.refs]
        )

        return GatheredLayers(
            refs=LayerRefs(layer_refs_uniq),
            gather=gather,
        )

    def _for_input(self, batch: int, input: Input):
        match input:
            case Refs(_) as refs:
                return self._for_refs(batch, refs)
            case GatheredLayers():
                return input
            case _:
                assert False, f"{input}"

    def _for_layer_base(self, batch: int, base: LayerBase):
        match base:
            case InputLayerBase(input=input):
                input = self._for_input(batch, input)
                return InputLayerBase(input=input)
            case LinearLayerBase(input=input, weight=weight):
                input = self._for_input(batch, input)
                weight = self._for_input(batch, weight)
                return LinearLayerBase(input=input, weight=weight)
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather):
                input = self._for_input(batch, input)
                weight = self._for_input(batch, weight)
                return LinearGatherLayerBase(input=input, weight=weight, gather=gather)
            case _:
                assert False

    def concat_inputs_layers(self):
        for bid, batch in self.network.batches.items():
            for layer in batch.layers.values():
                layer.base = self._for_layer_base(bid, layer.base)


def concat_inputs_layers(network: VectorizedNetwork):
    ConcatInputsLayers(network).concat_inputs_layers()
    return network
