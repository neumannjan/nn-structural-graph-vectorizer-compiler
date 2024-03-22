import torch
from neuralogic.core.builder.builder import NeuralSample

from lib.benchmarks.runnables.runnable import Runnable
from lib.datasets.dataset import BuiltDatasetInstance
from lib.sources import from_java
from lib.nn.topological.network_module import NetworkModule
from lib.nn.definitions.settings import Settings


class NeuralogicVectorizedTorchRunnable(Runnable):
    def __init__(self, device: str, settings: Settings) -> None:
        self._device = device
        self.settings = settings

    def initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        if samples is None:
            self.samples = dataset.samples

        print("Layers discovery...")
        self.network = from_java(self.samples, self.settings)

        self.model = NetworkModule(self.network, self.settings)

        if self.settings.compilation == "trace":
            self.model = torch.jit.trace_module(self.model, {"forward": ()}, strict=False)
        elif self.settings.compilation == "script":
            self.model = torch.jit.script(self.model)
        self.model.to(self.device)

    def forward_pass(self):
        return self.model()

    @property
    def device(self):
        return self._device
