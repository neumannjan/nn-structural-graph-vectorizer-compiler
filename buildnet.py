import torch
from lib.benchmarks.runnables.neuralogic_vectorized import NeuralogicVectorizedTorchRunnable
from lib.datasets.tu_molecular import MyTUDataset
from lib.engines.torch.settings import TorchModuleSettings
from lib.sources.neuralogic_settings import NeuralogicSettings
from lib.vectorize.settings import (
    LinearsSymmetriesSettings,
    OptimizeSingleUseGathersSettings,
    OptimizeTailRefsSettings,
    VectorizeSettings,
)

if __name__ == "__main__":
    debug = False
    device = "cpu"

    n_settings = NeuralogicSettings()
    n_settings.iso_value_compression = False
    n_settings.chain_pruning = False

    t_settings = TorchModuleSettings()

    v_settings = VectorizeSettings(
        transpose_fixed_count_reduce=True,
        iso_compression=True,
        linears_optimize_unique_ref_pairs=True,
        linears_symmetries=LinearsSymmetriesSettings(
            pad="any",
        ),
        optimize_tail_refs=OptimizeTailRefsSettings(
            unique_margin_rate=0.05,
        ),
        optimize_single_use_gathers=OptimizeSingleUseGathersSettings(
            run_before_symmetries=False,
            margin=30,
            margin_rate=0.05,
            aggressive_max_chain_depth="unlimited",
            propagate_through_symmetries=True,
        ),
        allow_repeat_gathers=True,
        merge_trivial_layer_concats=True,
        max_nogather_simple_layer_refs_length=24,
        granularize_by_weight=False,
    )

    # dataset = MyMutagenesis(n_settings, "simple", "original")
    dataset = MyTUDataset(n_settings, "mutag", "gsage")

    print("Dataset:", dataset)
    print("Device:", device)
    print("Settings:", n_settings)

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

    runnable = NeuralogicVectorizedTorchRunnable(
        device=device,
        neuralogic_settings=n_settings,
        torch_settings=t_settings,
        vectorize_settings=v_settings,
        debug=debug,
    )
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
