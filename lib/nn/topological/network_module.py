import torch
from tqdm.auto import tqdm

from lib.datasets import MyMutagenesis
from lib.nn.sources.source import LayerType, NeuralNetworkDefinition
from lib.nn.topological.aggregation_layer import AggregationLayer
from lib.nn.topological.fact_layer import FactLayer
from lib.nn.topological.settings import Settings
from lib.nn.topological.weighted_atom_layer import WeightedAtomLayer
from lib.nn.topological.weighted_rule_layer import WeightedRuleLayer
from lib.nn.utils.pipes import LayerOutputPipe

d = MyMutagenesis()


class NetworkModule(torch.nn.Module):
    def __init__(
        self,
        source: NeuralNetworkDefinition,
        settings: Settings,
    ) -> None:
        super().__init__()
        model = torch.nn.Sequential()

        for l, neurons in tqdm(source.items(), desc="Layers"):
            if l.type == "FactLayer":
                module = FactLayer(neurons, assume_facts_same=settings.assume_facts_same)
            elif l.type == "WeightedAtomLayer":
                module = WeightedAtomLayer(neurons, optimize_linear_gathers=settings.optimize_linear_gathers)
            elif l.type == "WeightedRuleLayer":
                module = WeightedRuleLayer(neurons, optimize_linear_gathers=settings.optimize_linear_gathers)
            elif l.type == "AggregationLayer":
                module = AggregationLayer(neurons)
            else:
                raise NotImplementedError(l.type)

            model.append(LayerOutputPipe(layer=l.id, delegate=module))

        self.model = model

    def forward(self):
        return self.model(None)
