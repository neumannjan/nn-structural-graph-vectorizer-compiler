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
from lib.nn.utils.pipes import LayerOutputPipe

d = MyMutagenesis()


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
                )
            elif l.type == "AggregationNeuron":
                module = AggregationLayer(
                    network[l.index],
                    ordinals,
                )
            else:
                raise NotImplementedError(l.type)

            model.append(LayerOutputPipe(layer=l.index, delegate=module))

        self.model = model

    def forward(self):
        return self.model(None)
