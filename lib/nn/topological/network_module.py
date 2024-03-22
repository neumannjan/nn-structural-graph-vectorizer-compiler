import torch
from torch.jit import unused
from tqdm.auto import tqdm

from lib.nn.sources.base import Network
from lib.nn.sources.views.map_ordinals import MapOrdinalsView
from lib.nn.sources.views.merge_facts import MergeFactsView
from lib.nn.topological.layer import Layer
from lib.nn.topological.settings import Settings

DEBUG_LAYERS = False


class NetworkModule(torch.nn.Module):
    def __new__(
        cls,
        network: Network,
        settings: Settings,
    ):
        if DEBUG_LAYERS:
            return super().__new__(_NetworkModuleWithTryBlocks)

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
        layer_shapes: dict[str, list[int]] = {}

        for l in tqdm(layers):
            neurons = network[l]
            layer_module = Layer.from_network(l.id, network, neurons, layer_shapes, settings)

            if settings.optimize_tail_gathers and l != layers[-1]:
                layer_module, ord_map = layer_module.unwrap_final_gather(layer_shapes)
                if len(ord_map) > 0:
                    network = MapOrdinalsView(network, ord_map)

            layer_shapes[str(l.id)] = layer_module.compute_output_shape(layer_shapes)
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
        self.layer_values: dict[str, torch.Tensor] = {}

    def forward(self):
        try:
            self.layer_values = {}
            for l, module in zip(self.layer_ids, self.model):
                self.layer_values = module(self.layer_values)
            return self.layer_values
        except Exception as e:
            raise Exception(f"Exception in layer {l}") from e
