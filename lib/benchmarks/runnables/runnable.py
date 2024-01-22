from typing import Protocol, Sequence

from neuralogic.core.builder.builder import NeuralSample

from lib.datasets.dataset import BuiltDatasetInstance


class Runnable(Protocol):
    def initialize(self, dataset: BuiltDatasetInstance, samples: Sequence[NeuralSample] | None = None):
        ...

    def forward_pass(self):
        ...

    @property
    def device(self):
        ...
