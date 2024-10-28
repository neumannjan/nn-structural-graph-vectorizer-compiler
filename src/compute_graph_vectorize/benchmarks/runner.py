from tqdm.auto import tqdm

from compute_graph_vectorize.benchmarks.runnables.runnable import Runnable
from compute_graph_vectorize.benchmarks.utils.timer import Timer, TimerResult
from compute_graph_vectorize.datasets.dataset import BuiltDatasetInstance


def measure_forward(runnable: Runnable, dataset: BuiltDatasetInstance, times: int) -> TimerResult:
    timer = Timer(runnable.device)

    runnable.initialize(dataset)

    for _ in tqdm(range(times), desc="Runs"):
        runnable.measure_forward_pass_epoch(timer)

    return timer.get_result()


def measure_backward(
    runnable: Runnable, dataset: BuiltDatasetInstance, times: int
) -> tuple[TimerResult, TimerResult, TimerResult]:
    fwd_timer = Timer(runnable.device)
    bwd_timer = Timer(runnable.device)
    all_timer = Timer(runnable.device)

    runnable.initialize(dataset)

    for _ in tqdm(range(times), desc="Runs"):
        runnable.measure_forward_and_backward_pass_epoch(fwd_timer, bwd_timer, all_timer)

    return fwd_timer.get_result(), bwd_timer.get_result(), all_timer.get_result()
