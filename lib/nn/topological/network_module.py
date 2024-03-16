import torch
from torch.jit import unused
from tqdm.auto import tqdm

from lib.datasets import MyMutagenesis
from lib.nn.sources.base import Network
from lib.nn.sources.views.merge_facts import MergeFactsView
from lib.nn.topological.layer import Layer
from lib.nn.topological.settings import Settings

d = MyMutagenesis()


def _build_model(
    network: Network,
    settings: Settings,
):
    model = torch.nn.ModuleList()

    if settings.merge_same_facts:
        network = MergeFactsView(network)

    for l, neurons in tqdm(network.items(), desc="Layers"):
        layer_module = Layer(l.id, network, neurons, settings)
        model.append(layer_module)

    return model


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
        self.model = _build_model(network, settings)

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

    def forward(self):
        try:
            layer_values: dict[str, torch.Tensor] = {}
            for module in self.model:
                layer_values = module(layer_values)
            return layer_values
        except Exception as e:
            raise Exception(f"Exception in layer {l.id}") from e
