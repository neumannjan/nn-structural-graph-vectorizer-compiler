from compute_graph_vectorize.sources.base import (
    LayerDefinitions,
    LayerNeurons,
    Network,
    Neurons,
    Ordinals,
    WeightDefinition,
)
from compute_graph_vectorize.utils import print_with_ellipsis


class BaseOrdinals(Ordinals):
    def __repr__(self) -> str:
        vals = (f"({lo.layer}, {lo.ordinal})" for lo in self)

        return f"{self.__class__.__name__}({print_with_ellipsis(iter(vals))} (length: {len(self)}))"


class BaseLayerDefinitions(LayerDefinitions):
    def __repr__(self) -> str:
        vals = ", ".join((str(ld.id) for ld in self))
        return f"{self.__class__.__name__}({vals})"


class BaseWeightDefinition(WeightDefinition):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, learnable={self.learnable})"


class BaseNeurons(Neurons):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length: {len(self)})"


class BaseLayerNeurons(LayerNeurons):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.layer.id}: {self.layer.type} -> (length: {len(self)}))"


class BaseNetwork(Network):
    def __repr__(self) -> str:
        items = [f"({ns.layer.id}: {ns.layer.type} -> (length: {len(ns)}))" for ns in self]
        items = ", ".join(items)

        return f"{self.__class__.__name__}({items})"
