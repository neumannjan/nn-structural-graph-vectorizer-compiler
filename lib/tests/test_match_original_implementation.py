import itertools

import jpype
import pytest
import torch
from lib.benchmarks.runnables.torch_gather_runnable import TorchGatherRunnable
from lib.datasets.dataset import MyDataset
from lib.datasets.mutagenesis import MyMutagenesis, MyMutagenesisMultip
from lib.nn.topological.settings import Settings
from lib.tests.utils.test_params import DEVICE_PARAMS, SETTINGS_PARAMS

DATASET_PARAMS = [
    MyMutagenesis(),
    MyMutagenesisMultip(),
]


def do_test_dataset(dataset: MyDataset, device: str, settings: Settings):
    try:
        dataset.settings.compute_neuron_layer_indices = True
        # dataset.settings.iso_value_compression = False
        # dataset.settings.chain_pruning = False
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

        runnable = TorchGatherRunnable(device=device, settings=settings)
        runnable.initialize(built_dataset_inst)

        print(runnable.model)

        results: dict[int, torch.Tensor] = runnable.forward_pass()

        for layer in runnable.network.layers:
            expected = torch.squeeze(torch.stack(list(runnable.network[layer.id].get_values_torch())))
            actual = torch.squeeze(results[layer.id]).detach().cpu()
            assert (torch.abs(expected - actual) < tolerance).all(), (
                f"Values do not match at layer {layer.id} ({layer.type}). "
                f"Max difference is {torch.max(torch.abs(expected - actual))}. "
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

        print("Expected:", expected)
        print("Actual:", actual)
        print("All values match!")

    except jpype.JException as e:
        print(e.message())
        print(e.stacktrace())
        raise e


@pytest.mark.parametrize(["device", "settings"], list(itertools.product(DEVICE_PARAMS, SETTINGS_PARAMS)))
def test_mutagenesis(device: str, settings: Settings):
    do_test_dataset(MyMutagenesis(), device, settings)


@pytest.mark.parametrize(["device", "settings"], list(itertools.product(DEVICE_PARAMS, SETTINGS_PARAMS)))
@pytest.mark.long
def test_mutagenesis_multip(device: str, settings: Settings):
    do_test_dataset(MyMutagenesisMultip(), device, settings)


if __name__ == "__main__":
    stts = SETTINGS_PARAMS[0]
    stts.optimize_linear_gathers = True
    stts.group_learnable_weight_parameters = False
    test_mutagenesis("cpu", stts)
