from typing import Sequence

from neuralogic.core.builder.builder import NeuralSample

from lib.benchmarks.runnables.runnable import Runnable
from lib.datasets.dataset import BuiltDatasetInstance


class NeuraLogicCPURunnable(Runnable):
    def initialize(self, dataset: BuiltDatasetInstance, samples: Sequence[NeuralSample] | None = None):
        self.neuralogic = dataset.neuralogic

        if samples is None:
            samples = dataset.samples

        self.samples = samples

    def forward_pass(self):
        self.neuralogic(self.samples)

    @property
    def device(self):
        return 'cpu'
