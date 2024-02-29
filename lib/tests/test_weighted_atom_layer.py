import itertools
import math
from typing import Sequence

import numpy as np
import pytest
import torch
from lib.nn.sources.dict_source import NeuralNetworkDefinitionDict, Neuron, WeightDefinitionImpl
from lib.nn.sources.source import LayerDefinition
from lib.nn.topological.settings import Settings
from lib.nn.topological.weighted_atom_layer import WeightedAtomLayer
from lib.tests.utils.test_params import SETTINGS_PARAMS
from lib.utils import atleast_3d_rev

LAYERS = [
    LayerDefinition(16, "FactLayer"),
    LayerDefinition(13, "WeightedAtomLayer"),
    LayerDefinition(12, "AggregationLayer"),
]

WEIGHTS: dict[int, list[float]] = {
    13: [0.39, -0.34, 0.99],
    5: [-0.75, 0.43, 0.33],
    0: [-0.62, 0.9, -0.66],
    8: [0.72, 0.56, 0.88],
    6: [0.47, 0.36, -0.33],
    1: [1, 0.46, -0.47],
    9: [-0.43, -0.77, -0.95],
    -1: [1.0],
    -2: [1.0],
}

OUTPUTS: dict[int, list[float]] = {
    13: [0.3714, -0.3275, 0.7574],
    5: [-0.6351, 0.4053, 0.3185],
    0: [-0.5511, 0.7163, -0.5784],
    8: [0.6169, 0.5080, 0.7064],
    6: [0.4382, 0.3452, -0.3185],
    1: [0.7616, 0.4301, -0.4382],
    9: [-0.4053, -0.6469, -0.7398],
    -1: [math.tanh(1.0)] * 3,
    -2: [math.tanh(1.0)],
}


def build_sample_from_input_indices(indices: Sequence[int]) -> tuple[NeuralNetworkDefinitionDict, torch.Tensor]:
    weights = [
        WeightDefinitionImpl(id=i, value=np.expand_dims(torch.tensor(WEIGHTS[i]), -1), learnable=i >= 0)
        for i in indices
    ]

    sample = NeuralNetworkDefinitionDict(
        layers=LAYERS,
        neurons={
            16: [Neuron(0)],
            13: [Neuron(1000 + i, inputs=[0], weights=[w]) for i, w in enumerate(weights)],
            12: [Neuron(1000 + len(weights), inputs=[1000 + i for i in range(len(weights))])],
        },
    )

    outputs = torch.tensor([OUTPUTS[i] for i in indices]).unsqueeze(-1)

    return sample, outputs


INDICES_PARAMS = [
    [13, 5, 0, 8, 6, 1, 9],
    [13, 5, 0, 8, 6, 1, 9, -1],
    [-1, 13, 5, 0, 8, 6, 1, 9],
    [13, 5, 0, -1, 8, 6, 1, 9],
    [-2],
]


@pytest.mark.parametrize(["indices", "settings"], list(itertools.product(INDICES_PARAMS, SETTINGS_PARAMS)))
def test_weighted_atom_layer(indices: Sequence[int], settings: Settings):
    inputs = {
        16: atleast_3d_rev(torch.tensor([1.0])),
    }

    network, expected = build_sample_from_input_indices(indices)

    layer = WeightedAtomLayer(network, network[13], settings=settings)

    print(layer)

    actual = layer(inputs)

    print("expected", expected.squeeze())
    print("actual", actual.squeeze())
    print("expected shape", expected.shape)
    print("actual shape", actual.shape)

    assert ((expected - actual).abs() <= 1e-4).all()


if __name__ == "__main__":
    test_weighted_atom_layer(INDICES_PARAMS[0], SETTINGS_PARAMS[0])
