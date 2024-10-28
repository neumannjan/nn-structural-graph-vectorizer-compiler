import torch
import torch.nn.functional as F
from neuralogic.core.builder.builder import NeuralSample

from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.utils.timer import Timer
from lib.datasets.dataset import BuiltDatasetInstance
from lib.engines.torch.from_vectorized import build_torch_network, simple_forward_pass_runner
from lib.engines.torch.settings import TorchModuleSettings
from lib.sources import from_neuralogic
from lib.sources.neuralogic_settings import NeuralogicSettings
from lib.vectorize.model.op_network import VectorizedOpSeqNetwork
from lib.vectorize.settings import VectorizeSettings


class PrebuiltNeuralogicVectorizedTorchRunnable(Runnable):
    def __init__(
        self,
        device: str,
        vectorized_network: VectorizedOpSeqNetwork,
        torch_settings: TorchModuleSettings,
        debug: bool,
    ) -> None:
        super().__init__()
        self._device = device
        self.t_settings = torch_settings
        self.vectorized_network = vectorized_network
        self.debug = debug

    def _initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        if samples is None:
            self.samples = dataset.samples

        self.targets = torch.tensor([s.target for s in self.samples], dtype=torch.get_default_dtype()).to(self.device)

        yield "Building PyTorch modules..."
        self.model = build_torch_network(self.vectorized_network, debug=self.debug, settings=self.t_settings)

        if self.t_settings.compilation == "trace":
            self.model = torch.jit.trace_module(self.model, {"forward": ()}, strict=False)
        elif self.t_settings.compilation == "script":
            self.model = torch.jit.script(self.model)
        self.model.to(self.device)  # pyright: ignore
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # pyright: ignore
        self.model.train()  # pyright: ignore

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

    def measure_forward_pass_epoch(self, timer: Timer):
        assert timer.device == self.device

        with timer:
            self.model()  # pyright: ignore

    def measure_forward_and_backward_pass_epoch(
        self,
        forward_timer: Timer,
        backward_timer: Timer,
        combined_timer: Timer,
    ):
        assert forward_timer.device == self.device
        assert backward_timer.device == self.device
        assert combined_timer.device == self.device

        self.optimizer.zero_grad()
        with combined_timer:
            with forward_timer:
                out = self.model()  # pyright: ignore
            loss = F.mse_loss(out.squeeze(), self.targets.squeeze())  # pyright: ignore
            with backward_timer:
                loss.backward()
            self.optimizer.step()

    @property
    def device(self):
        return self._device


class NeuralogicVectorizedTorchRunnable(PrebuiltNeuralogicVectorizedTorchRunnable):
    def __init__(
        self,
        device: str,
        neuralogic_settings: NeuralogicSettings,
        vectorize_settings: VectorizeSettings,
        torch_settings: TorchModuleSettings,
        debug: bool,
    ) -> None:
        self.n_settings = neuralogic_settings
        from lib.vectorize.pipeline.pipeline import create_vectorized_network_compiler

        self.build_vectorized_network = create_vectorized_network_compiler(
            vectorize_settings,
            forward_pass_runner=simple_forward_pass_runner,
            debug_prints=debug,
        )

        super().__init__(device, torch_settings=torch_settings, vectorized_network=None, debug=debug)
        self.t_settings = torch_settings
        self.v_settings = vectorize_settings
        self.debug = debug

    def _initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        if samples is None:
            self.samples = dataset.samples

        yield "Retrieving input information..."
        self.network = from_neuralogic(self.samples, self.n_settings)

        yield "Vectorizing..."
        self.vectorized_network = self.build_vectorized_network(self.network)

        yield from super()._initialize(dataset, samples)
