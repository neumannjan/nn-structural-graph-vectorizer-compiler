import itertools

import pytest
import torch
from lib.nn.gather import TakeEachNth, TakeLayerSlice, TakeValue, build_optimal_gather_module
from lib.nn.topological.layers import LayerOrdinal, TopologicalNetwork, compute_neuron_ordinals
from lib.nn.topological.settings import Settings
from lib.nn.utils.pipes import LayerPipe
from lib.tests.utils.network_mock import generate_example_network
from lib.utils import atleast_3d_rev


def _do_the_test(
    gather_module: torch.nn.Module,
    input_layer_ordinal_pairs: list[LayerOrdinal],
    network: TopologicalNetwork,
    all_same: bool,
):
    # input: indices of the neurons (so that for each neuron, its index is in its position)
    layer_values = {l: atleast_3d_rev(torch.tensor([n.getIndex() for n in neurons])) for l, neurons in network.items()}

    # expected output: list of the neuron indices that the module is supposed to gather
    expected = torch.tensor([network[l][o].getIndex() for l, o in input_layer_ordinal_pairs])

    # actual output:
    actual: torch.Tensor = torch.squeeze(gather_module(layer_values))

    if all_same:
        # if all inputs are the same,
        # the gather module takes advantage of that, but doesn't expand it by default
        expected = expected[0]

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
    ["inputs_from_previous_layer_only", "assume_facts_same", "execution_number"],
    GATHER_TEST_PARAMS,
)
def test_gather_module(
    inputs_from_previous_layer_only: bool,
    assume_facts_same: bool,
    execution_number: int,
):
    settings = Settings(assume_facts_same=assume_facts_same)

    layers, network = generate_example_network(inputs_from_previous_layer_only=inputs_from_previous_layer_only)
    _, ordinals = compute_neuron_ordinals(layers, network, settings=settings)
    for l in layers[1:]:
        input_layer_ordinal_pairs = [ordinals[inp.getIndex()] for n in network[l.index] for inp in n.getInputs()]

        gather = build_optimal_gather_module(input_layer_ordinal_pairs)

        print("Layer", l.index)
        print(gather)
        _do_the_test(gather, input_layer_ordinal_pairs, network, all_same=(assume_facts_same and l == 1))


def _get_underlying_gather_module(gather_module: torch.nn.Module):
    while isinstance(gather_module, LayerPipe):
        gather_module = gather_module.delegate

    return gather_module


def test_slice_gather():
    slice_start = 10
    slice_end = 100
    ordinals = list(range(slice_start, slice_end))
    total = 400
    input_layer_ordinal_pairs = [LayerOrdinal(0, i) for i in ordinals]
    inputs = {0: torch.tensor(list(range(total)))}
    expected = torch.tensor(ordinals)

    gather_module = build_optimal_gather_module(input_layer_ordinal_pairs)
    actual = gather_module(inputs)

    assert isinstance(_get_underlying_gather_module(gather_module), TakeLayerSlice)
    assert (actual == expected).all()


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
    input_layer_ordinal_pairs = [LayerOrdinal(0, idx)]
    inputs = {0: _unsqueeze_times(torch.arange(0, total, dtype=torch.int), times=unsqueeze_times)}

    gather_module = build_optimal_gather_module(input_layer_ordinal_pairs)
    actual = gather_module(inputs)

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
    input = torch.arange(0, range_len, dtype=torch.float).repeat(range_repeats)

    input_layer_ordinal_pairs = [LayerOrdinal(0, o) for o in ordinals_to_take]

    inputs = {0: input}

    gather_module = build_optimal_gather_module(input_layer_ordinal_pairs)
    actual = gather_module(inputs)

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
    input = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=torch.float)

    input_layer_ordinal_pairs = [LayerOrdinal(0, o) for o in ordinals_to_take]

    inputs = {0: input}

    gather_module = build_optimal_gather_module(input_layer_ordinal_pairs)
    actual = gather_module(inputs)
    expected_t = torch.tensor(expected, dtype=torch.float)

    print("Expected:", expected_t, "shape:", expected_t.shape)
    print("Actual:", actual, "shape:", actual.shape, flush=True)

    assert isinstance(_get_underlying_gather_module(gather_module), TakeEachNth)
    assert (actual == expected_t).all()
