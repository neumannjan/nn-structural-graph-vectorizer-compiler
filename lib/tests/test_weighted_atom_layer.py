import itertools
import math
from typing import Sequence

import numpy as np
import pytest
import torch
from lib.nn.topological.layers import LayerDefinition, compute_neuron_ordinals, get_neurons_per_layer
from lib.nn.topological.settings import Settings
from lib.nn.topological.weighted_atom_layer import WeightedAtomLayer
from lib.tests.utils.network_mock import MockJavaNeuron, MockJavaWeight
from lib.tests.utils.test_params import SETTINGS_PARAMS
from lib.utils import atleast_3d_rev

LAYERS = [
    LayerDefinition("", 16),
    LayerDefinition("", 13),
    LayerDefinition("", 12),
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
    6: [ 0.4382,  0.3452, -0.3185],
    1: [ 0.7616,  0.4301, -0.4382],
    9: [-0.4053, -0.6469, -0.7398],
    -1: [math.tanh(1.0)] * 3,
    -2: [math.tanh(1.0)],
}


def build_sample_from_input_indices(indices: Sequence[int]) -> tuple[MockJavaNeuron, torch.Tensor]:
    index_factory = iter(itertools.count())

    def _n():
        return next(index_factory)

    input_neuron = MockJavaNeuron(_n(), 12)

    weights = [
        MockJavaWeight(index=i, value=np.expand_dims(np.array(WEIGHTS[i]), -1), learnable=i >= 0) for i in indices
    ]

    sample = MockJavaNeuron(
        _n(),
        16,
        [  #
            MockJavaNeuron(_n(), 13, inputs=[input_neuron], weights=[w]) for w in weights
        ],
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
        12: atleast_3d_rev(torch.tensor([1.0])),
    }

    sample, expected = build_sample_from_input_indices(indices)
    network = get_neurons_per_layer([sample])

    _, ordinals = compute_neuron_ordinals(LAYERS, network, settings)

    layer = WeightedAtomLayer(sample.getInputs(), ordinals)

    print(layer)

    actual = layer(inputs)

    print("expected", expected.squeeze())
    print("actual", actual.squeeze())
    print("expected shape", expected.shape)
    print("actual shape", actual.shape)

    assert ((expected - actual).abs() <= 1e-4).all()


if __name__ == "__main__":
    test_weighted_atom_layer(INDICES_PARAMS[0], SETTINGS_PARAMS[0])
