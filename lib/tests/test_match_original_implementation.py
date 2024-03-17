import itertools

import jpype
import pytest
import torch
from lib.benchmarks.runnables.neuralogic_vectorized import NeuralogicVectorizedTorchRunnable
from lib.datasets.dataset import MyDataset
from lib.datasets.mutagenesis import MyMutagenesis
from lib.datasets.tu_molecular import MyTUDataset
from lib.nn.topological.settings import Settings
from lib.tests.utils.test_params import DEVICE_PARAMS, SETTINGS_PARAMS
from torch_geometric.data.dataset import warnings
from torch_geometric.datasets.citation_full import Callable

DatasetConstructor = Callable[[Settings], MyDataset]


COMMON_DATASET_PARAMS: list[DatasetConstructor] = [
    lambda settings: MyTUDataset(settings, "mutag", "gcn"),
    lambda settings: MyTUDataset(settings, "mutag", "gsage"),
]

EXTENDED_DATASET_PARAMS: list[DatasetConstructor] = [
    lambda settings: MyMutagenesis(settings, "simple", "original"),
    lambda settings: MyMutagenesis(settings, "simple_nobond", "original"),
]

LONG_DATASET_PARAMS: list[DatasetConstructor] = [
    lambda settings: MyMutagenesis(settings, "simple", "10x"),
    lambda settings: MyMutagenesis(settings, "simple_nobond", "10x"),
]


class CustomError(Exception):
    def __init__(self, *args: object, network, model) -> None:
        super().__init__(*args)
        self.network = network
        self.model = model


def do_test_dataset(dataset: MyDataset, device: str, settings: Settings):
    if device == "mps" and settings.allow_non_builtin_torch_ops:
        warnings.warn("Skipping MPS test for 'allow_non_builtin_torch_ops=True'.")
        return None, None

    try:
        print("Building dataset...")
        built_dataset_inst = dataset.build(sample_run=True)

        ###### CONFIG ######

        if device == "mps":
            # MPS doesn't support float64
            # we lower the requirement on value tolerance because of this
            tolerance = 1e-3
            torch.set_default_dtype(torch.float32)
        else:
            tolerance = 1e-8
            torch.set_default_dtype(torch.float64)

        ###### ALGORITHM ######

        runnable = NeuralogicVectorizedTorchRunnable(device=device, settings=settings)
        runnable.initialize(built_dataset_inst)

        print(runnable.network)
        print(runnable.model)

        results: dict[str, torch.Tensor] = runnable.forward_pass()

        if settings.optimize_tail_gathers:
            # with this optimization, intermediate layers won't match anymore.
            layers_to_check = [runnable.network.layers.as_list()[-1]]
        else:
            # can safely check all layers
            layers_to_check = runnable.network.layers

        for layer in layers_to_check:
            if settings.merge_same_facts and layer.type == "FactLayer":
                continue

            expected = torch.squeeze(torch.stack(list(runnable.network[layer.id].get_values_torch())))
            actual = torch.squeeze(results[str(layer.id)]).detach().cpu()
            assert expected.shape == actual.shape, (
                f"Shapes do not match at layer {layer.id} ({layer.type}).\n"
                f"Expected: {expected.shape}\n"
                f"Actual: {actual.shape}"
            )

            assert (torch.abs(expected - actual) < tolerance).all(), (
                f"Values do not match at layer {layer.id} ({layer.type}). "
                f"Max difference is {torch.max(torch.abs(expected - actual))}. "
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

        print("Expected:", expected)
        print("Actual:", actual)
        print("All values match!")

        return runnable.model, runnable.network

    except jpype.JException as e:
        print(e.message())
        print(e.stacktrace())
        raise CustomError(network=runnable.network, model=runnable.model) from e
    except Exception as e:
        raise CustomError(network=runnable.network, model=runnable.model) from e


@pytest.mark.parametrize(
    ["dataset_constructor", "device", "settings"],
    list(itertools.product(COMMON_DATASET_PARAMS, DEVICE_PARAMS, SETTINGS_PARAMS)),
)
@pytest.mark.common
def test(dataset_constructor: DatasetConstructor, device: str, settings: Settings):
    do_test_dataset(dataset_constructor(settings), device, settings)


@pytest.mark.parametrize(
    ["dataset_constructor", "device", "settings"],
    list(itertools.product(EXTENDED_DATASET_PARAMS, DEVICE_PARAMS, SETTINGS_PARAMS)),
)
@pytest.mark.extended
def test_extended(dataset_constructor: DatasetConstructor, device: str, settings: Settings):
    do_test_dataset(dataset_constructor(settings), device, settings)


@pytest.mark.parametrize(
    ["dataset_constructor", "device", "settings"],
    list(itertools.product(LONG_DATASET_PARAMS, DEVICE_PARAMS, SETTINGS_PARAMS)),
)
@pytest.mark.long
def test_long(dataset_constructor: DatasetConstructor, device: str, settings: Settings):
    do_test_dataset(dataset_constructor(settings), device, settings)


if __name__ == "__main__":
    settings = SETTINGS_PARAMS[0]
    settings.optimize_tail_gathers = True
    settings.compilation = "none"
    settings.neuralogic.iso_value_compression = True
    dataset = MyTUDataset(settings, source="mutag", template="gcn")
    # dataset = MyMutagenesis(settings, source="original", template="simple")
    try:
        model, network = do_test_dataset(dataset, "cpu", settings)
    except CustomError as e:
        model, network = e.model, e.network
        raise e
