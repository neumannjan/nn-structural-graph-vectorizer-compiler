import torch
from neuralogic.core.builder.builder import NeuralSample

from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.utils.timer import Timer
from lib.datasets.dataset import BuiltDatasetInstance
from lib.engines.torch.from_vectorized import build_torch_network
from lib.nn.definitions.settings import Settings
from lib.nn.topological.network_module import NetworkModule
from lib.sources import from_java
from lib.vectorize.pipeline.pipeline import build_vectorized_network


class NeuralogicLegacyVectorizedTorchRunnable(Runnable):
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
        self.model.to(self.device)  # pyright: ignore

    def forward_pass(self):
        return self.model()  # pyright: ignore

    @property
    def device(self):
        return self._device


class NeuralogicVectorizedTorchRunnable(Runnable):
    def __init__(self, device: str, settings: Settings, debug: bool) -> None:
        self._device = device
        self.settings = settings
        self.debug = debug

    def _initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        if samples is None:
            self.samples = dataset.samples

        yield "Retrieving input information..."
        self.network = from_java(self.samples, self.settings)

        yield "Vectorizing..."
        self.vectorized_network = build_vectorized_network(self.network)

        yield "Building PyTorch modules..."
        self.model = build_torch_network(
            self.vectorized_network,
            debug=self.debug,
            allow_non_builtin_torch_ops=self.settings.allow_non_builtin_torch_ops,
        )

        if self.settings.compilation == "trace":
            self.model = torch.jit.trace_module(self.model, {"forward": ()}, strict=False)
        elif self.settings.compilation == "script":
            self.model = torch.jit.script(self.model)
        self.model.to(self.device)  # pyright: ignore

        yield "Done."

    def initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        timer: Timer | None = None
        for desc in self._initialize(dataset, samples):
            if timer is not None:
                timer.stop()
                print(timer.get_result())

            print(desc, end=" ", flush=True)
            timer = Timer(device="cpu", agg_skip_first=0)
            timer.start()

        if timer is not None:
            timer.stop()
            print(timer.get_result())

    def forward_pass(self):
        return self.model()  # pyright: ignore

    @property
    def device(self):
        return self._device
