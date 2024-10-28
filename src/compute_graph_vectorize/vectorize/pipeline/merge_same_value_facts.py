from typing import Sequence

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.utils.values import HashableArray

_Ref = tuple[str, int]


class MergeSameValueFacts:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _get_merged_fact_layer(
        self, layer_id: str, facts: Sequence[Fact], out_ref_remap: dict[_Ref, int]
    ) -> list[Fact]:
        new_facts: list[Fact] = []

        value_to_key_map: dict[HashableArray, int] = {}

        for i, fact in enumerate(facts):
            match fact:
                case ValueFact(value=value):
                    value = HashableArray(value)
                    key = value_to_key_map.get(value, None)

                    if key is None:
                        key = len(new_facts)
                        value_to_key_map[value] = key
                        new_facts.append(fact)

                    if i != key:
                        out_ref_remap[layer_id, i] = key
                case _:
                    key = len(new_facts)
                    new_facts.append(fact)
                    if i != key:
                        out_ref_remap[layer_id, i] = key

        return new_facts

    def _remap_refs(self, refs: Refs, ref_remap: dict[_Ref, int]):
        for i, (t, l, o) in enumerate(refs):
            if t == Refs.TYPE_FACT:
                o2 = ref_remap.get((l, o), None)

                if o2 is not None:
                    refs.ordinals[i] = o2

    def _remap_layer_base(self, layer_base: LayerBase, ref_remap: dict[_Ref, int]):
        match layer_base:
            case InputLayerBase(input=Refs() as input):
                self._remap_refs(input, ref_remap)
            case LinearLayerBase(input=Refs() as input, weight=Refs() as weight):
                self._remap_refs(input, ref_remap)
                self._remap_refs(weight, ref_remap)
            case LinearGatherLayerBase(input=Refs() as input, weight=Refs() as weight):
                self._remap_refs(input, ref_remap)
                self._remap_refs(weight, ref_remap)
            case _:
                assert False

    def merge_same_value_facts(self):
        # merge the facts
        ref_remap: dict[_Ref, int] = {}

        for id, fact_layer in self.network.fact_layers.items():
            fact_layer.facts = self._get_merged_fact_layer(id, fact_layer.facts, ref_remap)

        # rebuild all references
        for batch in self.network.batches.values():
            for layer in batch.layers.values():
                self._remap_layer_base(layer.base, ref_remap)


def merge_same_value_facts(network: VectorizedLayerNetwork):
    MergeSameValueFacts(network).merge_same_value_facts()
    return network
