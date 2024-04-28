import argparse
from collections import defaultdict
from typing import Any, Literal, OrderedDict, Sequence
from typing import get_args as t_get_args

from lib.benchmarks.runnables.neuralogic_cpu import NeuraLogicCPURunnable
from lib.benchmarks.runnables.neuralogic_vectorized import NeuralogicVectorizedTorchRunnable
from lib.benchmarks.runnables.pyg import PytorchGeometricRunnable
from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.runner import MultiRunner
from lib.datasets.datasets import add_parser_args_for_dataset, build_dataset, get_dataset_info_from_args
from lib.datasets.mutagenesis import MutagenesisSource, MutagenesisTemplate
from lib.engines.torch.settings import Compilation, TorchModuleSettings
from lib.sources.neuralogic_settings import NeuralogicSettings
from lib.vectorize.settings import VectorizeSettings
from tqdm.auto import tqdm

Device = Literal["mps", "cuda", "cpu"]
Model = Literal["neuralogic_java", "neuralogic_torch", "torch_geometric"]


DEVICE_SUPPORT_MTX: dict[Device, list[Model]] = defaultdict(
    lambda: ["neuralogic_torch", "torch_geometric"],
    {"cpu": ["neuralogic_java", "neuralogic_torch", "torch_geometric"]},
)


class CommaSeparatedListAction(argparse.Action):
    def __init__(
        self,
        choices: list[str] | None = None,
        nargs=None,
        metavar=None,
        *kargs,
        **kwargs,
    ) -> None:
        if nargs is not None:
            raise ValueError("nargs not allowed")

        if metavar is None and choices is not None:
            metavar = f"[{{{','.join(choices)}}}, ...]"

        super().__init__(nargs=1, choices=None, metavar=metavar, *kargs, **kwargs)
        self._choices = choices

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if values is None:
            raise ValueError(f"{option_string} must receive at least one value.")
        elif not isinstance(values, str):
            values = values[0]

        items = str(values).split(",")

        out = []

        for item in items:
            item = item.strip()
            if self._choices is not None and item not in self._choices:
                raise ValueError(
                    f"{option_string} does not support value {item}: not one of {', '.join(self._choices)}"
                )
            out.append(item)

        setattr(namespace, self.dest, out)


DEFAULT_TORCH_SETTINGS = TorchModuleSettings()
DEFAULT_NEURALOGIC_SETTINGS = NeuralogicSettings(compute_neuron_layer_indices=True)
DEFAULT_VECTORIZE_SETTINGS = VectorizeSettings()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_parser_args_for_dataset(parser)
    parser.add_argument("--devices", "-d", action=CommaSeparatedListAction, choices=t_get_args(Device), required=True)
    parser.add_argument("--models", "-m", action=CommaSeparatedListAction, choices=t_get_args(Model), required=True)
    parser.add_argument("--repeats", "-n", "-r", type=int, default=10)
    parser.add_argument("--results-dict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compilation", "-c", choices=t_get_args(Compilation), default=DEFAULT_TORCH_SETTINGS.compilation
    )
    parser.add_argument(
        "--iso", action=argparse.BooleanOptionalAction, default=DEFAULT_NEURALOGIC_SETTINGS.iso_value_compression
    )
    parser.add_argument(
        "--chain", action=argparse.BooleanOptionalAction, default=DEFAULT_NEURALOGIC_SETTINGS.chain_pruning
    )
    parser.add_argument(
        "--tail", action=argparse.BooleanOptionalAction, default=DEFAULT_VECTORIZE_SETTINGS.optimize_tail_refs
    )
    parser.add_argument(
        "--uniq",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_VECTORIZE_SETTINGS.linears_optimize_unique_ref_pairs,
    )
    args = parser.parse_args()

    v_settings = VectorizeSettings()
    v_settings.optimize_tail_refs = args.tail
    v_settings.linears_optimize_unique_ref_pairs = args.uniq

    t_settings = TorchModuleSettings()
    t_settings.compilation = args.compilation

    n_settings = NeuralogicSettings()
    n_settings.iso_value_compression = args.iso
    n_settings.chain_pruning = args.chain

    print("------")
    print()
    print("Config:")
    print()
    for k in vars(args):
        v = getattr(args, k)
        print(f"{k}: {v}")
    print()
    print("------")
    print()
    print()

    dataset_info = get_dataset_info_from_args(args)

    runnables: dict[tuple[Model, Device], Runnable] = OrderedDict()

    models: list[Model] = list(args.models)
    devices: set[Device] = set(args.devices)

    for device in devices:
        for model in DEVICE_SUPPORT_MTX[device]:
            if model in models:
                if model == "neuralogic_java":
                    assert device == "cpu"
                    runnables[model, device] = NeuraLogicCPURunnable()
                elif model == "torch_geometric":
                    runnables[model, device] = PytorchGeometricRunnable(device=device)
                elif model == "neuralogic_torch":
                    runnables[model, device] = NeuralogicVectorizedTorchRunnable(
                        device=device,
                        neuralogic_settings=n_settings,
                        torch_settings=t_settings,
                        vectorize_settings=v_settings,
                        debug=False,
                    )

    runner = MultiRunner(n_repeats=args.repeats)

    source: MutagenesisSource = args.source
    template: MutagenesisTemplate = args.template

    dataset = build_dataset(dataset_info, n_settings).build()

    with tqdm(runnables.keys(), desc="Runners") as p:
        for model, device in p:
            p.set_postfix(OrderedDict((("model", model), ("device", device))))
            runnable = runnables[model, device]
            runner.measure(f"{model}--{device}", runnable, dataset)

    for k, v in runner.get_result().items():
        print(f"{k}:", v)
    print()
    print("------")
    print()
    results = {}
    for k, v in runner.get_result().items():
        print(f"{k}:", v.times_s)
        results[k] = v.times_ns

    if args.results_dict:
        print(repr(results))
