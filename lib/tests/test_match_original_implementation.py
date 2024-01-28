import jpype
import pytest
import torch
from lib.benchmarks.runnables.torch_gather_runnable import TorchGatherRunnable
from lib.datasets.mutagenesis import MyMutagenesis
from lib.nn.topological.settings import Settings
from lib.utils import value_to_tensor
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available

TEST_PARAMS = [
    ["cpu"],
]

if is_cuda_available():
    TEST_PARAMS += [
        ["cuda"],
    ]

if is_mps_available():
    TEST_PARAMS += [
        ["mps"],
    ]


@pytest.mark.parametrize(["device"], TEST_PARAMS)
def test_mutagenesis(device: str):
    try:
        dataset = MyMutagenesis()
        dataset.settings.compute_neuron_layer_indices = True
        # dataset.settings.iso_value_compression = False
        # dataset.settings.chain_pruning = False
        print("Building dataset...")
        built_dataset_inst = dataset.build(sample_run=True)

        ###### CONFIG ######

        settings = Settings(
            # TODO assumptions
            check_same_layers_assumption=False,
        )

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

        for layer in runnable.layers:
            expected = torch.squeeze(
                torch.stack([value_to_tensor(n.getRawState().getValue()) for n in runnable.network[layer.index]])
            )
            actual = torch.squeeze(results[layer.index]).detach().cpu()
            assert (torch.abs(expected - actual) < tolerance).all(), (
                f"Values do not match at layer {layer.index} ({layer.type}). "
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
