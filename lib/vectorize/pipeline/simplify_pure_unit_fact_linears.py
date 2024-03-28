from lib.utils import cache
from lib.vectorize.model import *


class SimplifyPureUnitFactLinears:
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network

    @property
    @cache
    def _unit_fact_layers(self):
        _unit_fact_layers: list[str] = []

        for id, layer in self.network.fact_layers.items():
            if all((isinstance(f, UnitFact) for f in layer.facts)):
                _unit_fact_layers.append(id)

        return _unit_fact_layers

    def _is_unit_fact_ref(self, ref: Ref | LayerRef) -> bool:
        match ref:
            case FactRef(id=id, ordinal=o):
                return isinstance(self.network.fact_layers[id].facts[o], UnitFact)
            case FactLayerRef(id=id):
                return id in self._unit_fact_layers
            case _:
                return False

    def _is_unit_fact_refs(self, refs: Refs | LayerRefs) -> bool:
        return all((self._is_unit_fact_ref(r) for r in refs.refs))

    def _is_pure_unit_fact(self, input: Input) -> bool:
        match input:
            case Refs(_) as refs:
                return self._is_unit_fact_refs(refs)
            case GatheredLayers(refs=refs):
                return self._is_unit_fact_refs(refs)
            case _:
                assert False, f"{input}"

    def _add_gather_to_input(self, input: Input, gather2: Gather) -> Input:
        match input:
            case GatheredLayers(refs=LayerRefs(refs), gather=gather):
                raise NotImplementedError("Implement combining two gathers together")
            case Refs():
                raise NotImplementedError()

    def _for_layer_base(self, base: LayerBase):
        match base:
            case InputLayerBase():
                return base
            case LinearLayerBase(input=input, weight=weight):
                if self._is_pure_unit_fact(input):
                    return InputLayerBase(weight)
                else:
                    return base
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather):
                if self._is_pure_unit_fact(input):
                    return InputLayerBase(self._add_gather_to_input(weight, gather))
                else:
                    return base
            case _:
                assert False

    def simplify_pure_unit_fact_linears(self):
        for batch in self.network.batches.values():
            for layer in batch.layers.values():
                layer.base = self._for_layer_base(layer.base)


def simplify_pure_unit_fact_linears(network: VectorizedNetwork):
    SimplifyPureUnitFactLinears(network).simplify_pure_unit_fact_linears()
    return network
