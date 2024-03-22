import pytest
import torch
from lib import sources
from lib.sources.base import LayerDefinition
from lib.nn.topological.layer import Layer
from lib.nn.definitions.settings import Settings
from lib.tests.utils.neuron_factory import NeuronTestFactory
from lib.tests.utils.test_params import SETTINGS_PARAMS

LAYERS = [
    LayerDefinition(16, "FactLayer"),
    LayerDefinition(13, "AggregationLayer"),
    LayerDefinition(12, "AggregationLayer"),
]


def build_sample1():
    factory = NeuronTestFactory(layers=LAYERS, id_provider_starts=[0, 1000, 10000])

    return {
        16: [factory.create(16) for _ in range(12)],
        13: [
            factory.create(13, inputs=[0, 1, 2, 3], aggregation="sum"),
            factory.create(13, inputs=[4, 5, 6, 7], aggregation="sum"),
            factory.create(13, inputs=[8, 9, 10, 11], aggregation="sum"),
        ],
        12: [factory.create(12, inputs=[1000, 1001, 1002])],
    }


def build_sample2():
    factory = NeuronTestFactory(layers=LAYERS, id_provider_starts=[0, 1000, 10000])

    return {
        16: [factory.create(16) for _ in range(20)],
        13: [
            factory.create(13, inputs=[0, 1, 2, 3], aggregation="sum"),
            factory.create(13, inputs=[4, 5, 6], aggregation="sum"),
            factory.create(13, inputs=[7, 8, 9], aggregation="sum"),
            factory.create(13, inputs=[10, 11, 12, 13], aggregation="sum"),
            factory.create(13, inputs=[14, 15], aggregation="sum"),
            factory.create(13, inputs=[16, 17, 18, 19], aggregation="sum"),
        ],
        12: [factory.create(12, inputs=[1000, 1001, 1002, 1003, 1004, 1005])],
    }


@pytest.mark.parametrize("settings", SETTINGS_PARAMS)
def test_same_no_of_inputs(settings: Settings):
    inputs = {"16": torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5])}
    network = sources.from_dict(layers=LAYERS, neurons=build_sample1())
    layer = Layer.from_network(
        out_to=13,
        network=network,
        neurons=network[13],
        layer_shapes={16: len(inputs["16"])},
        settings=settings,
    )
    expected = torch.tensor([8, 12, 20])

    actual = layer(inputs)["13"]

    assert (expected == actual).all()


@pytest.mark.parametrize("settings", SETTINGS_PARAMS)
def test_variable_no_of_inputs(settings: Settings):
    inputs = {"16": torch.tensor([2, 2, 2, 2, 3, 3, 3, 5, 5, 5, 7, 7, 7, 7, 11, 11, 13, 13, 13, 13])}
    network = sources.from_dict(layers=LAYERS, neurons=build_sample2())
    layer = Layer.from_network(
        out_to=13,
        network=network,
        neurons=network[13],
        layer_shapes={16: len(inputs["16"])},
        settings=settings,
    )
    expected = torch.tensor([8, 9, 15, 28, 22, 52])

    actual = layer(inputs)["13"]

    assert (expected == actual).all()


if __name__ == "__main__":
    test_variable_no_of_inputs(SETTINGS_PARAMS[0])
