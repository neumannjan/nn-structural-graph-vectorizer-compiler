import torch
from torch.jit import unused
from tqdm.auto import tqdm

from lib.nn.sources.base import Network
from lib.nn.sources.views.map_ordinals import MapOrdinalsView
from lib.nn.sources.views.merge_facts import MergeFactsView
from lib.nn.topological.layer import Layer
from lib.nn.topological.settings import Settings


class NetworkModule(torch.nn.Module):
    def __new__(
        cls,
        network: Network,
        settings: Settings,
        debug_layers: bool = False,
    ):
        if debug_layers:
            return _NetworkModuleWithTryBlocks(network, settings)

        return super().__new__(cls)

    def __init__(
        self,
        network: Network,
        settings: Settings,
        debug_layers: bool = False,
    ) -> None:
        super().__init__()
        model = torch.nn.ModuleList()

        if settings.merge_same_facts:
            network = MergeFactsView(network)

        layers = network.layers.as_list()

        for l in tqdm(layers):
            neurons = network[l]
            layer_module = Layer.from_network(l.id, network, neurons, settings)

            if settings.optimize_tail_gathers and l != layers[-1]:
                tpl = layer_module.unwrap_final_gather()

                if tpl is not None:
                    layer_module2, ord_map = tpl
                    layer_module = layer_module2
                    network = MapOrdinalsView(network, ord_map)

            model.append(layer_module)

        self.model = model

    def forward(self):
        layer_values: dict[str, torch.Tensor] = {}
        for module in self.model:
            layer_values = module(layer_values)
        return layer_values

    def __getitem__(self, layer_ord: int):
        return self.model[layer_ord]


@unused
class _NetworkModuleWithTryBlocks(NetworkModule):
    def __init__(
        self,
        network: Network,
        settings: Settings,
    ) -> None:
        super().__init__(network, settings, debug_layers=True)
        self.layer_ids = [l.id for l in network.layers]

    def forward(self):
        try:
            layer_values: dict[str, torch.Tensor] = {}
            for l, module in zip(self.layer_ids, self.model):
                layer_values = module(layer_values)
            return layer_values
        except Exception as e:
            raise Exception(f"Exception in layer {l}") from e
