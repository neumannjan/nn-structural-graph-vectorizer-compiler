import torch
from lib.benchmarks.runnables.neuralogic_vectorized import NeuralogicVectorizedTorchRunnable
from lib.datasets.mutagenesis import MyMutagenesis
from lib.datasets.tu_molecular import MyTUDataset
from lib.nn.definitions.settings import Settings

if __name__ == "__main__":
    device = "cpu"
    settings = Settings()
    settings.neuralogic.iso_value_compression = False
    dataset = MyMutagenesis(settings, "simple", "original")
    # dataset = MyTUDataset(settings, "mutag", "gin")

    print("Dataset:", dataset)
    print("Device:", device)
    print("Settings:", settings)

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

    DEBUG = True

    ###### ALGORITHM ######

    runnable = NeuralogicVectorizedTorchRunnable(device=device, settings=settings, debug=DEBUG)
    runnable.initialize(built_dataset_inst)

    print(runnable.network)
    print(runnable.model)

    expected = torch.squeeze(
        torch.stack(list(runnable.network[runnable.network.layers.as_list()[-1]].get_values_torch()))
    )

    result: torch.Tensor = runnable.forward_pass()  # pyright: ignore
    actual = torch.squeeze(result.detach().cpu())

    assert expected.shape == actual.shape, (
        f"Shapes do not match.\n" f"Expected: {expected.shape}\n" f"Actual: {actual.shape}"
    )

    assert (torch.abs(expected - actual) < tolerance).all(), (
        f"Values do not match. "
        f"Max difference is {torch.max(torch.abs(expected - actual))}. "
        f"Expected: {expected}\n"
        f"Actual: {actual}"
    )

    print("Expected:", expected)
    print("Actual:", actual)
    print("All values match!")
