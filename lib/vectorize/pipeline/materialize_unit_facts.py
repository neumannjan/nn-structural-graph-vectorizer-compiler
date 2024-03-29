import itertools

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_shapes import ComputeLayerShapes


def _build_exact_unit_fact(shape: ConcreteShape) -> Fact:
    if len(shape) == 2 and shape[0] == shape[1]:
        return EyeFact(shape[0])

    return UnitFact()


class MaterializeUnitFacts:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._unit_layer_map: dict[ConcreteShape, str] = {}
        self._compute_shapes = ComputeLayerShapes(network)

    def _get_unit_layer_id(self, shape: ConcreteShape) -> str:
        if shape not in self._unit_layer_map:
            name = name_start = "unit_" + "_".join((str(v) for v in shape))

            if name in self.network.fact_layers:
                name_start += "__"
                for i in itertools.count():
                    name = name_start + str(i)

                    if name not in self.network.fact_layers:
                        break

            self._unit_layer_map[shape] = name
            return name

        return self._unit_layer_map[shape]

    def _for_input(self, batch: int, layer: str, input: GatheredLayers):
        shape = self._compute_shapes.compute_layer_refs_shape(batch, input.refs)

        if not isinstance(shape, ConcreteShape):
            raise Exception(f"Failed to materialize layer {layer} (batch {batch}): Found shape {shape}.")

        input.refs.facts = [ref if ref != "unit" else self._get_unit_layer_id(shape) for ref in input.refs.facts]

    def _for_layer_base(self, batch: int, layer: str, base: LayerBase) -> LayerBase:
        match base:
            case InputLayerBase(input=GatheredLayers() as input):
                self._for_input(batch, layer, input)
                return base
            case LinearLayerBase(input=GatheredLayers() as input, weight=GatheredLayers() as weight):
                self._for_input(batch, layer, input)
                self._for_input(batch, layer, weight)
                return base
            case LinearGatherLayerBase(
                input=GatheredLayers() as input,
                weight=GatheredLayers() as weight,
            ):
                self._for_input(batch, layer, input)
                self._for_input(batch, layer, weight)
                return base
            case _:
                assert False, f"{base}"

    def materialize_unit_facts(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                layer.base = self._for_layer_base(bid, lid, layer.base)

        if "unit" in self.network.fact_layers:
            del self.network.fact_layers["unit"]

        for shape, fact_layer_id in self._unit_layer_map.items():
            self.network.fact_layers[fact_layer_id] = FactLayer([_build_exact_unit_fact(shape)], count=1, shape=shape)


def materialize_unit_facts(network: VectorizedLayerNetwork):
    MaterializeUnitFacts(network).materialize_unit_facts()
    return network
