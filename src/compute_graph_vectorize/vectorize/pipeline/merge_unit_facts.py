from compute_graph_vectorize.vectorize.model import *


class MergeUnitFacts:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self._refs_to_replace: set[tuple[str, int]]
        self.network = network

    def _build_ref_map(self):
        refs_to_replace = set()

        to_delete: list[str] = []

        for id, fact_layer in self.network.fact_layers.items():
            all_unit = True
            for o, fact in enumerate(fact_layer.facts):
                match fact:
                    case UnitFact():
                        refs_to_replace.add((id, o))
                    case _:
                        all_unit = False

            if all_unit:
                to_delete.append(id)

        for id in to_delete:
            del self.network.fact_layers[id]

        self._refs_to_replace = refs_to_replace

    def _remap_refs(self, refs: Refs):
        for i, (t, l, o) in enumerate(zip(refs.types, refs.layer_ids, refs.ordinals)):
            if t == Refs.TYPE_FACT and (l, o) in self._refs_to_replace:
                refs.layer_ids[i] = "unit"
                refs.ordinals[i] = 0

    def _remap_layer_base(self, layer_base: LayerBase):
        match layer_base:
            case InputLayerBase(input=Refs() as input):
                self._remap_refs(input)
            case LinearLayerBase(input=Refs() as input, weight=Refs() as weight):
                self._remap_refs(input)
                self._remap_refs(weight)
            case LinearGatherLayerBase(input=Refs() as input, weight=Refs() as weight):
                self._remap_refs(input)
                self._remap_refs(weight)
            case _:
                assert False

    def merge_unit_facts(self):
        self._build_ref_map()

        if len(self._refs_to_replace) > 0:
            for batch in self.network.batches.values():
                for layer in batch.layers.values():
                    self._remap_layer_base(layer.base)

            self.network.fact_layers["unit"] = FactLayer([UnitFact()])


def merge_unit_facts(network: VectorizedLayerNetwork):
    MergeUnitFacts(network).merge_unit_facts()
    return network
