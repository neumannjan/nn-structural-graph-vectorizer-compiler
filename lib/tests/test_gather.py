import itertools
from typing import Sequence

import pytest
import torch
from lib.nn.gather import (
    GatherAndRepeat,
    GatherAndView,
    MultiLayerGather,
    SingleLayerGather,
    SliceValues,
    TakeEachNth,
    TakeValue,
    build_optimal_gather,
    build_optimal_multi_layer_gather,
)
from lib.sources.base import LayerOrdinal, Network
from lib.tests.utils.network_mock import generate_example_network
from lib.utils import atleast_3d_rev


def _do_the_test(
    gather_module: torch.nn.Module,
    network: Network,
    inputs_ordinals: Sequence[LayerOrdinal],
):
    # input: indices of the neurons (so that for each neuron, its index is in its position)
    layer_values = {str(ld.id): atleast_3d_rev(torch.tensor(list(neurons.ids))) for ld, neurons in network.items()}

    # expected output: list of the neuron indices that the module is supposed to gather
    ids_per_layer = {str(ld.id): list(network[ld].ordinals.ids()) for ld in network.layers}
    expected = torch.tensor([ids_per_layer[str(l)][o] for l, o in inputs_ordinals])

    # actual output:
    actual: torch.Tensor = torch.squeeze(gather_module(layer_values))

    print("Expected:", expected, "shape:", expected.shape)
    print("Actual:", actual, "shape:", actual.shape, flush=True)

    # assert
    assert (actual == expected).all()


GATHER_TEST_PARAMS = list(
    itertools.product(
        [False, True],
        [False, True],
        list(range(5)),
    )
)


@pytest.mark.parametrize(
    ["inputs_from_previous_layer_only", "use_unique_pre_gathers", "execution_number"],
    GATHER_TEST_PARAMS,
)
def test_gather_module(
    inputs_from_previous_layer_only: bool,
    use_unique_pre_gathers: bool,
    execution_number: int,
):
    network = generate_example_network(inputs_from_previous_layer_only=inputs_from_previous_layer_only)
    layer_shapes = {str(l.id): [len(ns), 1] for l, ns in network.items()}

    for ld, neurons in itertools.islice(network.items(), 1, None):
        inputs_ordinals = list(neurons.inputs.ordinals)

        gather = build_optimal_multi_layer_gather(
            inputs_ordinals,
            layer_shapes,
            use_unique_pre_gathers=use_unique_pre_gathers,
        )

        print("Layer", ld)
        print(gather)
        _do_the_test(gather, network, inputs_ordinals)


def _get_underlying_gather_module(gather_module):
    while True:
        if isinstance(gather_module, GatherAndRepeat):
            gather_module = gather_module.gather
        elif isinstance(gather_module, SingleLayerGather):
            gather_module = gather_module.delegate
        elif isinstance(gather_module, MultiLayerGather):
            gather_module = gather_module.final_gather
        elif isinstance(gather_module, GatherAndView):
            gather_module = gather_module.gather
        else:
            break

    return gather_module


def test_slice_gather():
    slice_start = 10
    slice_end = 100
    ordinals = list(range(slice_start, slice_end))
    total = 400
    inp = torch.tensor(list(range(total)))
    expected = torch.tensor(ordinals)

    gather_module = build_optimal_gather(ordinals)
    actual = gather_module(inp)

    assert isinstance(_get_underlying_gather_module(gather_module), SliceValues)
    assert (actual == expected).all()
    assert actual.shape == expected.shape


TAKE_PARAMS = [
    [0, 150, torch.tensor([150])],
    [1, 150, torch.tensor([[150]])],
    [2, 150, torch.tensor([[[150]]])],
]


def _unsqueeze_times(tensor: torch.Tensor, times: int) -> torch.Tensor:
    for _ in range(times):
        tensor = tensor.unsqueeze(-1)
    return tensor


@pytest.mark.parametrize(["unsqueeze_times", "idx", "expected"], TAKE_PARAMS)
def test_take(unsqueeze_times: int, idx: int, expected: torch.Tensor):
    total = 400
    ordinals = [idx]
    inp = _unsqueeze_times(torch.arange(0, total, dtype=torch.int), times=unsqueeze_times)

    gather_module = build_optimal_gather(ordinals)
    actual = gather_module(inp)

    assert isinstance(_get_underlying_gather_module(gather_module), TakeValue)
    assert (actual == expected).all()
    assert actual.shape == expected.shape


TAKE_EACH_NTH_PARAMS = [
    [0, 4, 4, [0, 4, 8, 12], 0.0],
    [1, 4, 4, [1, 5, 9, 13], 1.0],
    [2, 4, 4, [2, 6, 10, 14], 2.0],
    [3, 4, 4, [3, 7, 11, 15], 3.0],
]


@pytest.mark.parametrize(
    ["take_idx", "range_len", "range_repeats", "ordinals_to_take", "expected"], TAKE_EACH_NTH_PARAMS
)
def test_take_each_nth(take_idx: int, range_len: int, range_repeats: int, ordinals_to_take: list[int], expected: float):
    inp = torch.arange(0, range_len, dtype=torch.float).repeat(range_repeats)

    gather_module = build_optimal_gather(ordinals_to_take)
    actual = gather_module(inp)

    print("Expected:", expected)
    print("Actual:", actual, "shape:", actual.shape, flush=True)

    assert isinstance(_get_underlying_gather_module(gather_module), TakeEachNth)
    assert (actual == expected).all()


TAKE_EACH_NTH_NONDIV_PARAMS = [
    [[2, 6, 10], [2.0, 2.0, 2.0]],
    [[2, 6], [2.0, 2.0]],
]


@pytest.mark.parametrize(["ordinals_to_take", "expected"], TAKE_EACH_NTH_NONDIV_PARAMS)
def test_take_each_nth_when_total_length_not_divisible_by_width(ordinals_to_take: list[int], expected: list[float]):
    inp = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=torch.float)

    gather_module = build_optimal_gather(ordinals_to_take)
    actual = gather_module(inp)

    expected_t = torch.tensor(expected, dtype=torch.float)

    print("Expected:", expected_t, "shape:", expected_t.shape)
    print("Actual:", actual, "shape:", actual.shape, flush=True)

    assert isinstance(_get_underlying_gather_module(gather_module), TakeEachNth)
    assert (actual == expected_t).all()
