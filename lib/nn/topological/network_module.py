import torch
from tqdm.auto import tqdm

from lib.datasets import MyMutagenesis
from lib.nn.sources.base import Network
from lib.nn.sources.views.merge_facts import MergeFactsView
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
        network: Network,
        settings: Settings,
    ) -> None:
        super().__init__()
        model = torch.nn.Sequential()

        if settings.merge_same_facts:
            network = MergeFactsView(network)

        for l, neurons in tqdm(network.items(), desc="Layers"):
            if l.type == "FactLayer":
                module = FactLayer(neurons, settings)
            elif l.type == "WeightedAtomLayer":
                module = WeightedAtomLayer(network, neurons, settings)
            elif l.type == "WeightedRuleLayer":
                module = WeightedRuleLayer(network, neurons, settings)
            elif l.type == "AggregationLayer":
                module = AggregationLayer(neurons, settings)
            else:
                raise NotImplementedError(l.type)

            model.append(LayerOutputPipe(layer=l.id, delegate=module))

        self.model = model

    def forward(self):
        return self.model(None)

    def __getitem__(self, layer_ord: int):
        return self.model[layer_ord]
