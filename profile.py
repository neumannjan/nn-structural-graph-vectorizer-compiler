import argparse
from typing import Literal
from typing import get_args as t_get_args

from lib.benchmarks.runnables.neuralogic_cpu_runnable import NeuraLogicCPURunnable
from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.runnables.torch_gather_runnable import TorchGatherRunnable
from lib.datasets.mutagenesis import MyMutagenesis, MyMutagenesisMultip
from lib.nn.topological.settings import Settings
from torch import profiler

Device = Literal['mps', 'cuda', 'cpu', 'neuralogic']


SETTINGS = Settings(
    check_same_layers_assumption=False,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('device', choices=t_get_args(Device))
    parser.add_argument('--multip', '-m', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    runnables: dict[Device, Runnable] = {
        "mps": TorchGatherRunnable(device="mps", settings=SETTINGS),
        "cpu": TorchGatherRunnable(device="cpu", settings=SETTINGS),
        "cuda": TorchGatherRunnable(device="cuda", settings=SETTINGS),
        "neuralogic": NeuraLogicCPURunnable(),
    }

    if args.multip:
        dataset = MyMutagenesisMultip().build()
    else:
        dataset = MyMutagenesis().build()

    runnable = runnables[args.device]

    runnable.initialize(dataset)

    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
        runnable.forward_pass()

    prof.export_chrome_trace("./trace.json")
