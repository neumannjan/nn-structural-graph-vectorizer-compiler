import datetime
import itertools
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Container, Hashable, Literal
from typing import get_args as t_get_args

import click
import torch
from compute_graph_vectorize.benchmarks.runnables.neuralogic_cpu import NeuraLogicCPURunnable
from compute_graph_vectorize.benchmarks.runnables.neuralogic_vectorized import (
    NeuralogicVectorizedTorchRunnable,
    PrebuiltNeuralogicVectorizedTorchRunnable,
)
from compute_graph_vectorize.benchmarks.runnables.pyg import PytorchGeometricRunnable
from compute_graph_vectorize.benchmarks.runner import measure_backward, measure_forward
from compute_graph_vectorize.datasets.dataset import MyDataset
from compute_graph_vectorize.datasets.mutagenesis import MutagenesisSource, MutagenesisTemplate, MyMutagenesis
from compute_graph_vectorize.datasets.tu_molecular import MyTUDataset, TUDatasetSource, TUDatasetTemplate
from compute_graph_vectorize.engines.torch.settings import Compilation, TorchModuleSettings, TorchReduceMethod
from compute_graph_vectorize.sources.neuralogic_settings import NeuralogicSettings
from compute_graph_vectorize.utils import dataclass_to_shorthand, iter_empty, serialize_dataclass
from compute_graph_vectorize.vectorize.model.op_network import VectorizedOpSeqNetwork
from compute_graph_vectorize.vectorize.pipeline.other.reduce_op_network_value import (
    count_gather_items,
    count_gathers,
    sum_op_network_values,
)
from compute_graph_vectorize.vectorize.pipeline.other.replace_tensors_with_shapes import replace_tensors_with_shapes
from compute_graph_vectorize.vectorize.settings import VectorizeSettings
from compute_graph_vectorize.vectorize.settings_presets import VectorizeSettingsPresets, iterate_vectorize_settings_presets
from tqdm.std import tqdm

Device = Literal["mps", "cuda", "cpu", "ipu"]
Engine = Literal["java", "torch", "pyg"]


DEFAULT_TORCH_SETTINGS = TorchModuleSettings()
DEFAULT_NEURALOGIC_SETTINGS_VECTORIZE = NeuralogicSettings(
    # compute_neuron_layer_indices=True,
    iso_value_compression=False,
    chain_pruning=False,
)
DEFAULT_VECTORIZE_SETTINGS = VectorizeSettings()


class DatasetBuilder:
    def __init__(self, source, template, c: Callable[[NeuralogicSettings, Any, Any], MyDataset]) -> None:
        self.source = source
        self.template = template
        self._c = c

    def __call__(self, settings: NeuralogicSettings, /) -> MyDataset:
        return self._c(settings, self.source, self.template)


DATASET_OPTIONS: dict[str, tuple[DatasetBuilder, Container[Engine]]] = dict(
    (
        *[
            (f"mutag-{s}-{t}", (DatasetBuilder(s, t, MyMutagenesis), ("java", "torch")))
            for s, t in itertools.product(t_get_args(MutagenesisSource), t_get_args(MutagenesisTemplate))
        ],
        *[
            (f"tu-{s}-{t}", (DatasetBuilder(s, t, MyTUDataset), ("java", "torch", "pyg")))
            for s, t in itertools.product(t_get_args(TUDatasetSource), t_get_args(TUDatasetTemplate))
        ],
    )
)


@click.group()
def cli():
    pass


@dataclass(frozen=True)
class Variant:
    device: Device
    engine: Engine
    dataset: str
    backward: bool
    times: int

    @staticmethod
    def build(
        device: Device,
        engine: Engine,
        dataset: str,
        backward: bool,
        times: int,
        compilation: Compilation,
        reduce_method: TorchReduceMethod,
        settings: VectorizeSettings,
        iso: bool,
        chain: bool,
    ):
        _, allowed_engines = DATASET_OPTIONS[dataset]
        if engine not in allowed_engines:
            return None

        if engine == "java":
            if device != "cpu":
                return None

            return JavaVariant(device, engine, dataset, backward, times, iso, chain)
        elif engine == "pyg":
            return TorchVariant(device, engine, dataset, backward, times)
        elif engine == "torch":
            return VectorizedTorchVariant(
                device, engine, dataset, backward, times, compilation, reduce_method, settings
            )
        else:
            raise ValueError(engine)

    def serialize(self):
        return serialize_dataclass(self, call_self=False)

    @staticmethod
    def deserialize(d: dict[str, Any]):
        return Variant.build(
            d["device"],
            d["engine"],
            d["dataset"],
            d["backward"],
            d["times"],
            d.get("compilation", None),
            d.get("reduce_method", None),
            VectorizeSettings.deserialize(d["settings"]) if "settings" in d else None,  # pyright: ignore
            d.get("iso", None),
            d.get("chain", None),
        )


@dataclass(frozen=True)
class TorchVariant(Variant):
    pass


@dataclass(frozen=True)
class VectorizedTorchVariant(TorchVariant):
    compilation: Compilation
    reduce_method: TorchReduceMethod
    settings: VectorizeSettings


@dataclass(frozen=True)
class JavaVariant(Variant):
    iso: bool
    chain: bool


DEVICE_SUPPORT_MTX: set[tuple[Device, Engine]] = {
    ("mps", "torch"),
    ("mps", "pyg"),
    ("cuda", "torch"),
    ("cuda", "pyg"),
    ("ipu", "torch"),
    ("ipu", "pyg"),
    ("cpu", "java"),
    ("cpu", "torch"),
    ("cpu", "pyg"),
}


@cli.command()
@click.option("-h", "--device", "devices", multiple=True, type=click.Choice(t_get_args(Device)), required=True)
@click.option("-e", "--engine", "engines", multiple=True, type=click.Choice(t_get_args(Engine)), required=True)
@click.option(
    "-d", "--dataset", "datasets", multiple=True, type=click.Choice(list(DATASET_OPTIONS.keys())), required=True
)
@click.option("--backward", multiple=True, type=bool, default=(False,))
@click.option("-n", "--times", multiple=True, type=int, default=(10,))
@click.option(
    "-c", "--compilation", "compilations", multiple=True, type=click.Choice(t_get_args(Compilation)), default=("none",)
)
@click.option(
    "-r",
    "--reduce",
    "reduces",
    multiple=True,
    type=click.Choice(t_get_args(TorchReduceMethod)),
    default=("segment_csr",),
)
@click.option(
    "-s", "--settings", "settings_presets", type=click.Choice(t_get_args(VectorizeSettingsPresets)), default="tuning"
)
@click.option("--iso", multiple=True, type=bool, default=(True,))
@click.option("--chain", multiple=True, type=bool, default=(True,))
@click.argument("dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path))
def prepare(
    devices: tuple[Device, ...],
    engines: tuple[Engine, ...],
    datasets: tuple[str, ...],
    backward: tuple[bool, ...],
    times: tuple[int, ...],
    compilations: tuple[Compilation, ...],
    reduces: tuple[TorchReduceMethod, ...],
    settings_presets: VectorizeSettingsPresets,
    iso: tuple[bool, ...],
    chain: tuple[bool, ...],
    dir: Path,
):
    dir.mkdir(parents=True, exist_ok=True)
    file = dir / "variants.txt"

    if not iter_empty(itertools.islice(dir.iterdir(), 1, None)) and not file.exists():
        raise click.ClickException(f"Directory {dir.absolute()} is not empty.")

    device_engine_pairs = [p for p in itertools.product(devices, engines) if p in DEVICE_SUPPORT_MTX]
    print("Devices/Engines:", device_engine_pairs)
    print("Datasets:", datasets)
    print("Backward:", backward)
    print("Times:", times)
    print("Compilations:", compilations)
    print("Reduce methods:", reduces)
    print("ISO (Java only):", iso)
    print("Chain (Java only):", chain)

    all_variants = sorted(
        set(
            v
            for v in (
                Variant.build(h, e, d, b, t, c, r, s, i, ch)  # pyright: ignore
                for (h, e), d, b, t, c, r, s, i, ch in itertools.product(
                    device_engine_pairs,
                    datasets,
                    backward,
                    times,
                    compilations,
                    reduces,
                    iterate_vectorize_settings_presets(settings_presets),
                    iso,
                    chain,
                )  # pyright: ignore
            )
            if v is not None
        ),
        key=dataclass_to_shorthand,
    )

    print("Total:", len(all_variants))

    all_shorthands = [dataclass_to_shorthand(v) for v in all_variants]

    assert len(set(all_shorthands)) == len(all_shorthands)

    with open(file, "w") as fp:
        json.dump([v.serialize() for v in all_variants], fp, indent=2)


@cli.command()
@click.argument("dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
def total(dir: Path):
    file = dir / "variants.txt"

    if not file.exists():
        raise click.ClickException(f"{file.absolute()} does not exist.")

    with open(file, "r") as fp:
        variants = json.load(fp)
        print(len(variants))


@cli.command(context_settings={"show_default": True})
@click.argument(
    "dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path)
)
@click.argument("index", type=int)
@click.option("--measure/--no-measure", default=True)
@click.option("--save-architecture/--no-save-architecture", default=True)
@click.option("--force-cpu", default=False, is_flag=True)
def run(dir: Path, index: int, measure: bool, save_architecture: bool, force_cpu: bool):
    torch.set_default_dtype(torch.float32)

    variants_file = dir / "variants.txt"

    if not variants_file.exists():
        raise click.ClickException(f"{variants_file.absolute()} does not exist.")

    with open(variants_file, "r") as fp:
        variants = json.load(fp)

    if index < 0 or index >= len(variants):
        raise click.ClickException(
            f"Index for this directory must fit within the range from 0 (incl.) to {len(variants)} (excl.)"
        )

    variant = Variant.deserialize(variants[index])
    del variants

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    notimename = dataclass_to_shorthand(variant)
    timename = f"{notimename},{time}"

    print(variant)
    print()
    print(timename)
    print()

    device = "cpu" if force_cpu else variant.device

    if isinstance(variant, JavaVariant) and variant.engine == "java":
        runnable = NeuraLogicCPURunnable()
        dataset = DATASET_OPTIONS[variant.dataset][0](
            NeuralogicSettings(iso_value_compression=variant.iso, chain_pruning=variant.chain)
        )
    elif isinstance(variant, VectorizedTorchVariant) and variant.engine == "torch":
        runnable = NeuralogicVectorizedTorchRunnable(
            device=device,
            neuralogic_settings=DEFAULT_NEURALOGIC_SETTINGS_VECTORIZE,
            vectorize_settings=variant.settings,
            torch_settings=TorchModuleSettings(reduce_method=variant.reduce_method, compilation=variant.compilation),
            debug=False,
        )
        dataset = DATASET_OPTIONS[variant.dataset][0](DEFAULT_NEURALOGIC_SETTINGS_VECTORIZE)
    elif isinstance(variant, TorchVariant) and variant.engine == "pyg":
        runnable = PytorchGeometricRunnable(device=device)
        dataset = DATASET_OPTIONS[variant.dataset][0](DEFAULT_NEURALOGIC_SETTINGS_VECTORIZE)
    else:
        raise ValueError(variant)

    dataset = dataset.build()

    if measure:
        out_file = dir / f"{timename}.json"
        out = variant.serialize()

        if variant.backward:
            fwd, bwd, cmb = measure_backward(runnable, dataset, times=variant.times)
            out["fwd"] = fwd.times_ns.tolist()
            out["bwd"] = bwd.times_ns.tolist()
            out["cmb"] = cmb.times_ns.tolist()
            print("Forward: ", fwd)
            print("Backward:", bwd)
            print("Combined:", cmb)
        else:
            cmb = measure_forward(runnable, dataset, times=variant.times)
            out["cmb"] = out["fwd"] = cmb.times_ns.tolist()
            print("Forward:", cmb)

        with open(out_file, "w") as fp:
            json.dump(out, fp)
    elif save_architecture:
        runnable.initialize(dataset)

    if save_architecture and hasattr(runnable, "vectorized_network"):
        out_pkl_file = dir / f"{notimename}.pkl"
        print(out_pkl_file)
        # TODO
        with open(out_pkl_file, "wb") as fp:
            pickle.dump(runnable.vectorized_network, fp)


@cli.command(context_settings={"show_default": True})
@click.argument(
    "dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path)
)
@click.argument("index", type=int)
def run_prebuilt(dir: Path, index: int):
    torch.set_default_dtype(torch.float32)

    variants_file = dir / "variants.txt"

    if not variants_file.exists():
        raise click.ClickException(f"{variants_file.absolute()} does not exist.")

    with open(variants_file, "r") as fp:
        variants = json.load(fp)

    if index < 0 or index >= len(variants):
        raise click.ClickException(
            f"Index for this directory must fit within the range from 0 (incl.) to {len(variants)} (excl.)"
        )

    variant = Variant.deserialize(variants[index])
    del variants

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    notimename = dataclass_to_shorthand(variant)
    timename = f"{notimename},{time}"

    vectorized_network_file = dir / f"{notimename}.pkl"

    with open(vectorized_network_file, "rb") as fp:
        vectorized_network: VectorizedOpSeqNetwork = pickle.load(fp)

    print(variant)
    print()
    print(timename)
    print()

    device = variant.device

    assert isinstance(variant, VectorizedTorchVariant)

    runnable = PrebuiltNeuralogicVectorizedTorchRunnable(
        device=device,
        vectorized_network=vectorized_network,
        torch_settings=TorchModuleSettings(reduce_method=variant.reduce_method, compilation=variant.compilation),
        debug=False,
    )
    dataset = DATASET_OPTIONS[variant.dataset][0](DEFAULT_NEURALOGIC_SETTINGS_VECTORIZE)
    dataset = dataset.build()

    out_file = dir / f"{timename}.json"
    out = variant.serialize()

    if variant.backward:
        fwd, bwd, cmb = measure_backward(runnable, dataset, times=variant.times)
        out["fwd"] = fwd.times_ns.tolist()
        out["bwd"] = bwd.times_ns.tolist()
        out["cmb"] = cmb.times_ns.tolist()
        print("Forward: ", fwd)
        print("Backward:", bwd)
        print("Combined:", cmb)
    else:
        cmb = measure_forward(runnable, dataset, times=variant.times)
        out["cmb"] = out["fwd"] = cmb.times_ns.tolist()
        print("Forward:", cmb)

    with open(out_file, "w") as fp:
        json.dump(out, fp)


ArchMapMethod = Literal["exact", "inexact", "gather_total", "gather_counts", "gather"]


def _get_network_key(vectorized_network: VectorizedOpSeqNetwork, method: ArchMapMethod):
    if method == "exact":
        return vectorized_network
    elif method == "inexact":
        return replace_tensors_with_shapes(vectorized_network)
    elif method == "gather_total":
        return sum_op_network_values(vectorized_network, count_gathers)
    elif method == "gather_counts":
        return sum_op_network_values(vectorized_network, count_gather_items)
    elif method == "gather":
        a = sum_op_network_values(vectorized_network, count_gathers)
        b = sum_op_network_values(vectorized_network, count_gather_items)
        return (a, b)
    else:
        raise ValueError(method)


@cli.command(context_settings={"show_default": True})
@click.argument(
    "dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path)
)
@click.option("--method", type=click.Choice(choices=t_get_args(ArchMapMethod)), default="exact")
def build_architecture_map(dir: Path, method: ArchMapMethod):
    variants_file = dir / "variants.txt"

    if not variants_file.exists():
        raise click.ClickException(f"{variants_file.absolute()} does not exist.")

    with open(variants_file, "r") as fp:
        variants = json.load(fp)

    architectures: list[Hashable] = []
    architectures_dict: dict[Hashable, int] = {}
    variants_dict: dict[str, int] = {}

    for v in tqdm(variants):
        variant = Variant.deserialize(v)
        notimename = dataclass_to_shorthand(variant)
        pkl_file_path = dir / (notimename + ".pkl")
        if pkl_file_path.exists():
            with open(pkl_file_path, "rb") as fp:
                vectorized_network = pickle.load(fp)

            key = _get_network_key(vectorized_network, method)

            if key in architectures_dict:
                idx = architectures_dict[key]
            else:
                idx = len(architectures)
                # architectures.append(vectorized_network)
                architectures.append(key)
                architectures_dict[key] = idx

            variants_dict[notimename] = idx

    with open(dir / "networks.pkl", "wb") as fp:
        pickle.dump((architectures, variants_dict), fp)

    print(len(architectures))


@cli.command(context_settings={"show_default": True})
@click.argument(
    "dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path)
)
@click.argument("file", type=str)
def find_idx(dir: Path, file: str):
    variants_file = dir / "variants.txt"
    the_file = dir / file

    if not variants_file.exists():
        raise click.ClickException(f"{variants_file.absolute()} does not exist.")

    if not the_file.exists():
        raise click.ClickException(f"{the_file.absolute()} does not exist.")

    with open(variants_file, "r") as fp:
        variants = json.load(fp)

    with open(the_file, "r") as fp:
        result_data = json.load(fp)

    variant = Variant.deserialize(result_data)

    for i, vrt_d in enumerate(variants):
        vrt = Variant.deserialize(vrt_d)
        if vrt == variant:
            print(i)
            return

    print("Not found.")


if __name__ == "__main__":
    cli()
