import numpy as np

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation


class ConcatInputsLayers(LayerwiseOperation):
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network

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
        layer_refs = LayerRefs(
            facts=sorted(set((ref.id for ref in refs.refs if isinstance(ref, FactRef)))),
            weights=sorted(set((ref.id for ref in refs.refs if isinstance(ref, WeightRef)))),
            layers=sorted(set((ref.id for ref in refs.refs if isinstance(ref, NeuronRef)))),
        )
        layer_counts = list(ComputeLayerCounts(self.network).iter_layer_refs_counts(batch, layer_refs))
        layer_offsets: list[int] = np.concatenate([[0], np.cumsum(layer_counts[:-1], dtype=int)], dtype=int).tolist()

        fact_offset_map: dict[str, int] = {
            id: o for id, o in zip(layer_refs.facts, layer_offsets[: len(layer_refs.facts)])
        }
        weight_offset_map: dict[str, int] = {
            id: o
            for id, o in zip(
                layer_refs.weights,
                layer_offsets[len(layer_refs.facts) : len(layer_refs.facts) + len(layer_refs.weights)],
            )
        }
        layer_offset_map: dict[str, int] = {
            id: o
            for id, o in zip(
                layer_refs.layers,
                layer_offsets[len(layer_refs.facts) + len(layer_refs.weights) :],
            )
        }

        def get_layer_offset(ref: Ref):
            match ref:
                case FactRef(id=id, ordinal=_):
                    return fact_offset_map[id]
                case WeightRef(id=id):
                    return weight_offset_map[id]
                case NeuronRef(id=id, ordinal=_):
                    return layer_offset_map[id]
                case _:
                    assert False, f"{ref}"

        gather = GenericGather([get_layer_offset(ref) + self._get_ref_offset(ref) for ref in refs.refs])

        return GatheredLayers(refs=layer_refs, gather=gather)

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

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        layer.base = self._for_layer_base(batch, layer.base)
        return layer

    def concat_inputs_layers(self):
        for bid, batch in self.network.batches.items():
            for layer in batch.layers.values():
                layer.base = self._for_layer_base(bid, layer.base)


def concat_inputs_layers(network: VectorizedNetwork):
    ConcatInputsLayers(network).concat_inputs_layers()
    return network
