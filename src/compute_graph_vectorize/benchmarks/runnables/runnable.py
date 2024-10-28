from typing import Protocol

from neuralogic.core.builder.builder import NeuralSample

from compute_graph_vectorize.benchmarks.utils.timer import Timer, TimerResult
from compute_graph_vectorize.datasets.dataset import BuiltDatasetInstance


class Runnable(Protocol):
    def initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None): ...

    def forward_pass(self): ...

    def measure_forward_pass_epoch(self, timer: Timer): ...

    def measure_forward_and_backward_pass_epoch(self, forward_timer: Timer, backward_timer: Timer, combined_timer: Timer): ...

    @property
    def device(self): ...
