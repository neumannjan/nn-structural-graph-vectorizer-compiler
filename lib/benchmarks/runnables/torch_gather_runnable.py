from neuralogic.core.builder.builder import NeuralSample

from lib.benchmarks.runnables.runnable import Runnable
from lib.datasets.dataset import BuiltDatasetInstance
from lib.nn.topological.layers import compute_neuron_ordinals, discover_all_layers, get_neurons_per_layer
from lib.nn.topological.network_module import NetworkModule
from lib.nn.topological.settings import Settings


class TorchGatherRunnable(Runnable):
    def __init__(self, device: str, settings: Settings) -> None:
        self._device = device
        self.settings = settings

    def initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        if samples is None:
            self.samples = dataset.samples

        print("Layers discovery...")
        self.layers = discover_all_layers(self.samples, self.settings)

        self.network = get_neurons_per_layer(self.samples)

        _, self.ordinals = compute_neuron_ordinals(self.layers, self.network, self.settings)

        self.model = NetworkModule(self.layers, self.network, self.ordinals, self.settings)
        self.model.to(self._device)

    def forward_pass(self):
        return self.model()

    def device(self):
        return self._device
