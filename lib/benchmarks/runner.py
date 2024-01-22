import copy
from typing import Sequence

from neuralogic.core.builder.builder import NeuralSample
from tqdm.auto import tqdm

from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.utils.timer import Timer, TimerResult
from lib.datasets.dataset import BuiltDatasetInstance


class Runner:
    def __init__(self, n_repeats: int) -> None:
        self.n_repeats = n_repeats

    def measure(
        self, runnable: Runnable, dataset: BuiltDatasetInstance, samples: Sequence[NeuralSample] | None = None
    ) -> TimerResult:
        timer = Timer(runnable.device)

        runnable.initialize(dataset, samples)

        for _ in tqdm(range(self.n_repeats), desc="Runs"):
            with timer:
                runnable.forward_pass()

        return timer.get_result()


class MultiRunner:
    def __init__(self, n_repeats: int) -> None:
        self.n_repeats = n_repeats
        self._result: dict[str, TimerResult] = {}

    def measure(
        self,
        name: str,
        runnable: Runnable,
        dataset: BuiltDatasetInstance,
        samples: Sequence[NeuralSample] | None = None,
    ):
        runner = Runner(self.n_repeats)
        self._result[name] = runner.measure(runnable, dataset, samples)

    def get_result(self) -> dict[str, TimerResult]:
        return copy.deepcopy(self._result)
