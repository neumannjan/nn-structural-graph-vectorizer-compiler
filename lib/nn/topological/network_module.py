from typing import Sequence

import torch
from lib.datasets import MyMutagenesis
from lib.nn.topological.aggregation_layer import AggregationLayer
from lib.nn.topological.fact_layer import FactLayer
from lib.nn.topological.layers import (
    LayerDefinition,
    Ordinals,
    TopologicalNetwork,
)
from lib.nn.topological.settings import Settings
from lib.nn.topological.weighted_atom_layer import WeightedAtomLayer
from lib.nn.topological.weighted_rule_layer import WeightedRuleLayer
from lib.other_utils import camel_to_snake

d = MyMutagenesis()


class LayerPipe(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, layer_index: int) -> None:
        super().__init__()
        self.layer = layer
        self.layer_index = layer_index

    def forward(self, layer_values: dict[int, torch.Tensor] | None = None):
        # TODO: autodetect in preprocessing which layers can be thrown away when for saving memory
        if layer_values is None:
            layer_values = {}

        with torch.profiler.record_function(camel_to_snake(self.layer.__class__.__name__, upper=True)):
            layer_values[self.layer_index] = self.layer(layer_values)
        return layer_values

    def extra_repr(self) -> str:
        return f"layer_index={self.layer_index},"


class NetworkModule(torch.nn.Module):
    def __init__(
        self,
        layers: Sequence[LayerDefinition],
        network: TopologicalNetwork,
        ordinals: Ordinals,
        settings: Settings,
    ) -> None:
        super().__init__()
        model = torch.nn.Sequential()

        for l in layers:
            print()
            print(f"Layer {l.index}:")
            if l.type == "FactNeuron":
                module = FactLayer(network[l.index], assume_facts_same=settings.assume_facts_same)
            elif l.type == "WeightedAtomNeuron":
                module = WeightedAtomLayer(
                    network[l.index],
                    ordinals,
                )
            elif l.type == "WeightedRuleNeuron":
                module = WeightedRuleLayer(
                    network[l.index],
                    ordinals,
                    assume_rule_weights_same=settings.assume_rule_weights_same,
                    check_same_inputs_dim_assumption=settings.check_same_inputs_dim_assumption,
                )
            elif l.type == "AggregationNeuron":
                module = AggregationLayer(
                    network[l.index],
                    ordinals,
                )
            else:
                raise NotImplementedError(l.type)

            model.append(LayerPipe(module, layer_index=l.index))

        self.model = model

    def forward(self):
        return self.model(None)
