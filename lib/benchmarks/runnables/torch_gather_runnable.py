from typing import Sequence

from neuralogic.core.builder.builder import NeuralSample

from lib.benchmarks.runnables.runnable import Runnable
from lib.datasets.dataset import BuiltDatasetInstance
from lib.nn.topological.layers import compute_neuron_ordinals, discover_all_layers, get_neurons_per_layer
from lib.nn.topological.network_module import NetworkModule
from lib.nn.topological.settings import Settings


class TorchGatherRunnable(Runnable):
    def __init__(self, device: str) -> None:
        self._device = device

    def initialize(self, dataset: BuiltDatasetInstance, samples: Sequence[NeuralSample] | None = None):
        settings = Settings(
            # TODO assumptions
            check_same_layers_assumption=False,
        )

        if samples is None:
            samples = dataset.samples

        print("Layers discovery...")
        layers = discover_all_layers(samples, settings)

        network = get_neurons_per_layer(samples)

        _, ordinals = compute_neuron_ordinals(layers, network, settings)

        self.model = NetworkModule(layers, network, ordinals, settings)
        print(self.model)
        self.model.to(self._device)

    def forward_pass(self):
        self.model()

    def device(self):
        return self._device
