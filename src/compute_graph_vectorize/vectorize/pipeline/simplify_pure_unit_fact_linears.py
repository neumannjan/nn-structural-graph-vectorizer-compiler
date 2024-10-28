from compute_graph_vectorize.utils import cache
from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.layerwise import LayerwiseOperation


class SimplifyPureUnitFactLinears(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    @property
    @cache
    def _unit_fact_layers(self):
        _unit_fact_layers: list[str] = []

        for id, layer in self.network.fact_layers.items():
            if all((isinstance(f, UnitFact) for f in layer.facts)):
                _unit_fact_layers.append(id)

        return _unit_fact_layers

    def _is_unit_fact(self, fact: Fact) -> bool:
        match fact:
            case UnitFact():
                return True
            case EyeFact():
                return True
            case _:
                return False

    def _is_unit_fact_refs(self, refs: Refs) -> bool:
        return all((t == Refs.TYPE_FACT for t in refs.types)) and all(
            (self._is_unit_fact(self.network.fact_layers[l].facts[o]) for l, o in zip(refs.layer_ids, refs.ordinals))
        )

    def _is_unit_fact_layer_refs(self, refs: LayerRefs) -> bool:
        return all((t == Refs.TYPE_FACT for t in refs.types)) and all(
            (id in self._unit_fact_layers for id in refs.layer_ids)
        )

    def _is_pure_unit_fact(self, input: Input) -> bool:
        match input:
            case Refs() as refs:
                return self._is_unit_fact_refs(refs)
            case GatheredLayers(refs=LayerRefs() as refs):
                return self._is_unit_fact_layer_refs(refs)
            case _:
                assert False, f"{input}"

    def _add_gather_to_input(self, input: Input, _: Gather) -> Input:
        match input:
            case GatheredLayers(refs=LayerRefs(), gather=_):
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
                elif self._is_pure_unit_fact(weight):
                    return InputLayerBase(input)
                else:
                    return base
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather):
                if self._is_pure_unit_fact(input):
                    return InputLayerBase(self._add_gather_to_input(weight, gather))
                elif self._is_pure_unit_fact(weight):
                    return InputLayerBase(self._add_gather_to_input(input, gather))
                else:
                    return base
            case _:
                assert False

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        layer.base = self._for_layer_base(layer.base)
        return layer

    def simplify_pure_unit_fact_linears(self):
        for batch in self.network.batches.values():
            for layer in batch.layers.values():
                layer.base = self._for_layer_base(layer.base)


def simplify_pure_unit_fact_linears(network: VectorizedLayerNetwork):
    SimplifyPureUnitFactLinears(network).simplify_pure_unit_fact_linears()
    return network
