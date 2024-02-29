import argparse
from typing import Literal
from typing import get_args as t_get_args
import warnings

from lib.benchmarks.runnables.neuralogic_cpu_runnable import NeuraLogicCPURunnable
from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.runnables.torch_gather_runnable import TorchGatherRunnable
from lib.benchmarks.runner import MultiRunner
from lib.datasets.mutagenesis import MutagenesisSource, MutagenesisTemplate, MyMutagenesis
from lib.nn.topological.settings import Settings
from tqdm.auto import tqdm

Device = Literal['mps', 'cuda', 'cpu', 'neuralogic']


SETTINGS = Settings(
    check_same_layers_assumption=False,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multip', '-m', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--devices', '-d', choices=t_get_args(Device), nargs='+', required=True)
    parser.add_argument('--source', '-s', choices=t_get_args(MutagenesisSource), default="original")
    parser.add_argument('--template', '-t', choices=t_get_args(MutagenesisTemplate), default="simple")
    args = parser.parse_args()

    runnables: dict[Device, Runnable] = {
        "mps": TorchGatherRunnable(device="mps", settings=SETTINGS),
        "cpu": TorchGatherRunnable(device="cpu", settings=SETTINGS),
        "cuda": TorchGatherRunnable(device="cuda", settings=SETTINGS),
        "neuralogic": NeuraLogicCPURunnable(),
    }

    runner = MultiRunner(n_repeats=10)

    source: MutagenesisSource = args.source
    template: MutagenesisTemplate = args.template

    if args.multip:
        warnings.warn("--multip flag is deprecated. use `--source 10x` or `-s 10x` instead.")

        source = "10x"

    dataset = MyMutagenesis(template=template, source=source).build()

    for runnable_name in tqdm(args.devices, desc="Runners"):
        runnable = runnables[runnable_name]
        runner.measure(runnable_name, runnable, dataset)

    print(runner.get_result())
