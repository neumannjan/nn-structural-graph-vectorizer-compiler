from lib.vectorize.model import *


class MergeUnitFacts:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self._ref_map: dict[Ref, Ref]
        self.network = network
        self._ref = network.ref_pool.fact("unit", 0)

    def _build_ref_map(self):
        self._ref_map = {}

        to_delete: list[str] = []

        for id, fact_layer in self.network.fact_layers.items():
            all_unit = True
            for o, fact in enumerate(fact_layer.facts):
                match fact:
                    case UnitFact():
                        self._ref_map[self.network.ref_pool.fact(id=id, ordinal=o)] = self._ref
                    case _:
                        all_unit = False

            if all_unit:
                to_delete.append(id)

        for id in to_delete:
            del self.network.fact_layers[id]

        return self._ref_map

    def _remap_refs(self, refs: Refs):
        for i in range(len(refs.refs)):
            ref = refs.refs[i]
            refs.refs[i] = self._ref_map.get(ref, ref)

    def _remap_layer_base(self, layer_base: LayerBase):
        match layer_base:
            case InputLayerBase(input=Refs(_) as input):
                self._remap_refs(input)
            case LinearLayerBase(input=Refs(_) as input, weight=Refs(_) as weight):
                self._remap_refs(input)
                self._remap_refs(weight)
            case LinearGatherLayerBase(input=Refs(_) as input, weight=Refs(_) as weight):
                self._remap_refs(input)
                self._remap_refs(weight)
            case _:
                assert False

    def merge_unit_facts(self):
        self._build_ref_map()

        if len(self._ref_map) > 0:
            for batch in self.network.batches.values():
                for layer in batch.layers.values():
                    self._remap_layer_base(layer.base)

            self.network.fact_layers["unit"] = FactLayer([UnitFact()])


def merge_unit_facts(network: VectorizedLayerNetwork):
    MergeUnitFacts(network).merge_unit_facts()
    return network
