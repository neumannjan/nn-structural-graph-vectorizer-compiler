import pytest
import torch
from lib.nn.sources.dict_source import NeuralNetworkDefinitionDict
from lib.nn.sources.source import LayerDefinition
from lib.nn.topological.aggregation_layer import AggregationLayer
from lib.nn.topological.settings import Settings
from lib.tests.utils.network_mock import Neuron
from lib.tests.utils.test_params import SETTINGS_PARAMS

LAYERS = [
    LayerDefinition(16, "FactLayer"),
    LayerDefinition(13, "AggregationLayer"),
    LayerDefinition(12, "AggregationLayer"),
]

SAMPLE1 = {
    16: [Neuron(j) for j in range(12)],
    13: [
        Neuron(1000, [0, 1, 2, 3]),
        Neuron(1001, [4, 5, 6, 7]),
        Neuron(1002, [8, 9, 10, 11]),
    ],
    12: [Neuron(10000, [1000, 1001, 1002])],
}

SAMPLE2 = {
    16: [Neuron(j) for j in range(20)],
    13: [
        Neuron(1000, [0, 1, 2, 3]),
        Neuron(1001, [4, 5, 6]),
        Neuron(1002, [7, 8, 9]),
        Neuron(1003, [10, 11, 12, 13]),
        Neuron(1004, [14, 15]),
        Neuron(1005, [16, 17, 18, 19]),
    ],
    12: [Neuron(10000, [1000, 1001, 1002, 1003, 1004, 1005])],
}


@pytest.mark.parametrize("settings", SETTINGS_PARAMS)
def test_same_no_of_inputs(settings: Settings):
    inputs = {16: torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5])}
    network = NeuralNetworkDefinitionDict(layers=LAYERS, neurons=SAMPLE1)
    layer = AggregationLayer(
        neurons=network[13],
        aggregation_type="sum",
        settings=settings,
    )
    expected = torch.tensor([8, 12, 20])

    actual = layer(inputs)

    assert (expected == actual).all()


@pytest.mark.parametrize("settings", SETTINGS_PARAMS)
def test_variable_no_of_inputs(settings: Settings):
    inputs = {16: torch.tensor([2, 2, 2, 2, 3, 3, 3, 5, 5, 5, 7, 7, 7, 7, 11, 11, 13, 13, 13, 13])}
    network = NeuralNetworkDefinitionDict(layers=LAYERS, neurons=SAMPLE2)
    layer = AggregationLayer(
        neurons=network[13],
        aggregation_type="sum",
        settings=settings,
    )
    expected = torch.tensor([8, 9, 15, 28, 22, 52])

    actual = layer(inputs)

    assert (expected == actual).all()


if __name__ == "__main__":
    test_variable_no_of_inputs(SETTINGS_PARAMS[0])
