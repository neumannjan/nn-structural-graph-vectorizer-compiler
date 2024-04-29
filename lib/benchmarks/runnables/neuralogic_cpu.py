from neuralogic.core.builder.builder import NeuralSample

from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.utils.timer import Timer
from lib.datasets.dataset import BuiltDatasetInstance


class NeuraLogicCPURunnable(Runnable):
    def initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        self.neuralogic = dataset.neuralogic

        if samples is None:
            samples = dataset.samples

        self.samples = samples

    def forward_pass(self):
        return self.neuralogic(self.samples)

    def measure_forward_pass_epoch(self, timer: Timer):
        assert timer.device == self.device
        with timer:
            self.neuralogic(self.samples)

    def measure_forward_and_backward_pass_epoch(
        self,
        forward_timer: Timer,
        backward_timer: Timer,
        combined_timer: Timer,
    ):
        assert combined_timer.device == self.device

        with combined_timer:
            self.neuralogic(self.samples, train=True)

    @property
    def device(self):
        return "cpu"
