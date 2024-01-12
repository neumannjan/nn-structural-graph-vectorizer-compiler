import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import jpype
import numpy as np
import torch
from lib.datasets import MyMutagenesis
from neuralogic.core.builder.builder import NeuralSample
from tqdm.auto import tqdm

d = MyMutagenesis()


def iter_start_neurons(samples: NeuralSample | Iterable[NeuralSample]) -> Iterable:
    all_samples = []
    if isinstance(samples, NeuralSample):
        all_samples = [samples]
    else:
        all_samples = samples

    return (s.java_sample.query.neuron for s in all_samples)


@dataclass(frozen=True)
class LayerDefinition:
    type: str
    index: int


def get_neuron_type(java_neuron):
    return str(java_neuron.getClass().getSimpleName())


def discover_layers(sample: NeuralSample) -> list[LayerDefinition]:
    out: list[LayerDefinition] = []

    neuron = sample.java_sample.query.neuron
    initial_layer: int = neuron.getLayer()
    neighbor_layers: dict[int, int] = {}
    layer_types: dict[int, str] = {}

    queue = deque([neuron])

    while len(queue) > 0:
        neuron = queue.popleft()
        layer = neuron.getLayer()
        layer_type = get_neuron_type(neuron)
        if layer in layer_types:
            assert layer_types[layer] == layer_type
        else:
            layer_types[layer] = layer_type

        layer_inputs = list(neuron.getInputs())

        if len(layer_inputs) == 0:
            break

        # find closest input layer
        if layer in neighbor_layers:
            neighbor_layers[layer] = min(min((inp.getLayer() for inp in layer_inputs)), neighbor_layers[layer])
        else:
            neighbor_layers[layer] = min((inp.getLayer() for inp in layer_inputs))

        # continue with all closest
        queue.extend((inp for inp in layer_inputs if inp.getLayer() == neighbor_layers[layer]))

    # output result
    layer = initial_layer
    out += [LayerDefinition(layer_types[layer], layer)]
    while True:
        if layer not in neighbor_layers:
            break

        layer = neighbor_layers[layer]
        out += [LayerDefinition(layer_types[layer], layer)]

    out.reverse()
    return out


def discover_all_layers(samples: Sequence[NeuralSample], layers_verification=True) -> tuple[LayerDefinition, ...]:
    layers = tuple(discover_layers(built_dataset.samples[0]))

    if layers_verification:
        for sample in tqdm(samples[1:], desc="Verifying layers"):
            assert tuple(discover_layers(sample)) == layers

    return layers


def get_neurons_per_layer(samples: Sequence[NeuralSample]) -> dict[int, list]:
    queue = deque((sample.java_sample.query.neuron for sample in samples))

    visited = set()

    out: dict[int, list] = defaultdict(lambda: [])

    while len(queue) > 0:
        neuron = queue.popleft()

        neuron_index = int(neuron.getIndex())
        if neuron_index in visited:
            continue

        visited.add(neuron_index)
        out[int(neuron.getLayer())].append(neuron)

        for inp in neuron.getInputs():
            inp_index = int(inp.getIndex())
            if inp_index not in visited:
                queue.append(inp)

    return out


class GatherModule(torch.nn.Module):
    def __init__(self, input_layer_ordinal_pairs: list[tuple[int, int]]) -> None:
        super().__init__()

        self.input_layer_ordinal_pairs = input_layer_ordinal_pairs
        self.layers = sorted(set((l for l, _ in input_layer_ordinal_pairs)), reverse=True)
        layers_map = {l: i for i, l in enumerate(self.layers)}

        if len(self.layers) == 1:
            per_layer_ordinals: dict[int, list[int]] = {self.layers[0]: [o for _, o in input_layer_ordinal_pairs]}
        else:
            per_layer_ordinals: dict[int, list[int]] = {
                layer: sorted(set((o for l, o in input_layer_ordinal_pairs if l == layer))) for layer in self.layers
            }

            layer_sizes = [len(per_layer_ordinals[layer]) for layer in self.layers]
            layer_prefixes = np.concatenate([[0], np.cumsum(layer_sizes)[:-1]])

            per_layer_ordinals2_map: dict[int, dict[int, int]] = {
                l: {o: i for i, o in enumerate(per_layer_ordinals[l])} for l in self.layers
            }

            concatenated_ordinals = [
                layer_prefixes[layers_map[l]] + per_layer_ordinals2_map[l][o] for l, o in input_layer_ordinal_pairs
            ]

            self.concatenated_ordinals = torch.tensor(concatenated_ordinals)
        self.per_layer_ordinals = {l: torch.tensor(ordinals) for l, ordinals in per_layer_ordinals.items()}

    def forward(self, layer_values: dict[int, torch.Tensor]):
        # if there is only one layer, skip the additional gather
        if len(self.layers) == 1:
            layer = self.layers[0]
            return torch.index_select(layer_values[layer], 0, self.per_layer_ordinals[layer])

        layer_inputs_needed = [
            torch.index_select(layer_values[layer], 0, self.per_layer_ordinals[layer]) for layer in self.layers
        ]
        layer_shape_hull = [
            -1,
            max((v.shape[1] for v in layer_inputs_needed)),
            max((v.shape[2] for v in layer_inputs_needed)),
        ]
        layer_inputs_needed = [v.expand(*layer_shape_hull) for v in layer_inputs_needed]
        layer_inputs_needed = torch.concatenate(layer_inputs_needed)

        out = torch.index_select(layer_inputs_needed, 0, self.concatenated_ordinals)
        return out


def test_gather_module(gather_module: GatherModule, network: dict[int, list]):
    # input: indices of the neurons (so that for each neuron, its index is in its position)
    layer_values = {l: atleast_3d_rev(torch.tensor([n.getIndex() for n in neurons])) for l, neurons in network.items()}

    # expected output: list of the neuron indices that the module is supposed to gather
    expected = torch.tensor([network[l][o].getIndex() for l, o in gather_module.input_layer_ordinal_pairs])

    # actual output:
    actual = torch.squeeze(gather_module(layer_values))

    # assert
    assert (actual == expected).all()


def value_to_numpy(java_value) -> np.ndarray:
    arr = np.asarray(java_value.getAsArray())
    arr = arr.reshape(java_value.size())
    return arr


def value_to_tensor(java_value) -> torch.Tensor:
    return torch.tensor(value_to_numpy(java_value))


def atleast_3d_rev(tensor: torch.Tensor) -> torch.Tensor:
    dim = tensor.dim()

    if dim == 0:
        return tensor.reshape([1, 1, 1])
    elif dim == 1:
        return tensor.reshape([-1, 1, 1])
    elif dim == 2:
        return tensor.reshape([*tensor.shape, 1])
    else:
        return tensor


def atleast_2d_rev(tensor: torch.Tensor) -> torch.Tensor:
    dim = tensor.dim()

    if dim == 0:
        return tensor.reshape([1, 1])
    elif dim == 1:
        return tensor.reshape([-1, 1])
    else:
        return tensor


def expand_diag(tensor: torch.Tensor, n: int) -> torch.Tensor:
    tensor = torch.squeeze(tensor)
    dim = tensor.dim()
    if dim > 2:
        raise ValueError()

    if dim == 2:
        return tensor

    if dim == 0:
        tensor = torch.atleast_1d(tensor).expand([n])

    return torch.diag(tensor)


class LayerPipe(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, layer_index: int) -> None:
        super().__init__()
        self.layer = layer
        self.layer_index = layer_index

    def forward(self, layer_values: dict[int, torch.Tensor] | None = None):
        # TODO: autodetect in preprocessing which layers can be thrown away when for saving memory
        if layer_values is None:
            layer_values = {}

        layer_values[self.layer_index] = self.layer(layer_values)
        return layer_values


class Linear(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        diagonal_expand: bool,
        assume_all_weights_same: bool,
    ) -> None:
        super().__init__()
        self.expand_diagonal = diagonal_expand

        if assume_all_weights_same:
            layer_neurons = layer_neurons[:1]

        self.weights_map = torch.nn.ParameterDict()
        self.all_weights = []

        self.weight_indices = []
        for weight in tqdm([w for n in layer_neurons for w in n.getWeights()], desc="Weights"):
            self.weight_indices += [weight.index]

            if str(weight.index) in self.weights_map:
                w_tensor = self.weights_map[str(weight.index)]
            else:
                w_tensor = value_to_tensor(weight.value)

                if weight.isLearnable:
                    w_tensor = torch.nn.Parameter(w_tensor)

                self.weights_map[str(weight.index)] = w_tensor

            self.all_weights += [w_tensor]

        if self.expand_diagonal:
            # ensure all weights are square matrices, vectors, or scalars
            for v in self.all_weights:
                assert v.dim() <= 2
                if torch.squeeze(v).dim() == 2:
                    assert v.shape[0] == v.shape[1]
        else:
            # ensure all weights are matrices
            for v in self.all_weights:
                assert v.dim() == 2

    def forward(self, input_values: torch.Tensor):
        w = self.all_weights
        if self.expand_diagonal:
            dim = max((torch.atleast_1d(torch.squeeze(v)).shape[0] for v in w))
            w = [expand_diag(v, dim) for v in w]
        else:
            # TODO: remove?
            w_shape_hull = [
                max((v.shape[0] for v in w)),
                max((v.shape[1] for v in w)),
            ]
            w = [v.expand(w_shape_hull) for v in w]

        w = torch.stack(w)
        y = w @ input_values
        return y


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: dict[int, tuple[int, int]],
        check_same_weights_assumption=True,
        check_same_inputs_dim_assumption=True,
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        neuron = layer_neurons[0]

        self.gather = GatherModule(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()]
        )

        self.linear = Linear(layer_neurons,
                             diagonal_expand=True,
                             assume_all_weights_same=True)

        self.inputs_dim = neuron.getInputs().size()

        if check_same_inputs_dim_assumption:
            for n in layer_neurons:
                assert self.inputs_dim == n.getInputs().size()

        if check_same_weights_assumption:
            layer_index = neuron.getLayer()

            for n in tqdm(layer_neurons[1:], desc="Verifying neurons"):
                assert n.getLayer() == layer_index

                for our_widx, weight in zip(self.linear.weight_indices, n.getWeights()):
                    assert our_widx == weight.index

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)
        input_values = torch.reshape(input_values, [-1, self.inputs_dim, *input_values.shape[1:]])

        y = self.linear(input_values)

        # TODO: parameterize
        y = torch.sum(y, 1)
        y = torch.tanh(y)
        return y


class WeightedAtomLayer(torch.nn.Module):
    def __init__(
        self, layer_neurons: list, neuron_ordinals: dict[int, tuple[int, int]], check_same_inputs_assumption=True
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        neuron = layer_neurons[0]

        # TODO: ASSUMPTION: all inputs are from the same layer
        # (easy fix: replace with GatherModule)
        input_layer_ordinal_pairs = [neuron_ordinals[int(n.getIndex())] for n in neuron.getInputs()]
        input_layers = set((l for l, _ in input_layer_ordinal_pairs))

        assert len(input_layers) == 1
        self.input_layer = next(iter(input_layers))
        self.input_ordinals = torch.tensor([o for _, o in input_layer_ordinal_pairs])

        self.linear = Linear(layer_neurons,
                             diagonal_expand=False,
                             assume_all_weights_same=False)

        if check_same_inputs_assumption:
            first_inputs = tuple((int(inp.getIndex()) for inp in neuron.getInputs()))
            for n in layer_neurons[1:]:
                this_inputs = tuple((int(inp.getIndex()) for inp in n.getInputs()))
                assert first_inputs == this_inputs

    def forward(self, layer_values: dict[int, torch.Tensor]):
        # TODO: replace with GatherModule?
        input_values = torch.index_select(layer_values[self.input_layer], 0, self.input_ordinals)
        # TODO reshape ?

        y = self.linear(input_values)
        y = torch.tanh(y)
        return y


class AggregationLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: dict[int, tuple[int, int]],
        check_same_inputs_dim_assumption=True,
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        self.gather = GatherModule(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()]
        )

        neuron = layer_neurons[0]
        self.inputs_dim = neuron.getInputs().size()

        if check_same_inputs_dim_assumption:
            for n in layer_neurons:
                assert self.inputs_dim == n.getInputs().size()

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)
        input_values = torch.reshape(input_values, [-1, self.inputs_dim, *input_values.shape[1:]])
        y = torch.mean(input_values, 1)
        return y


class FactLayer(torch.nn.Module):
    def __init__(self, layer_neurons: list, check_same_facts_assumption=True) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        neuron = layer_neurons[0]

        # TODO: ASSUMPTION: is not learnable
        assert not neuron.hasLearnableValue
        # TODO: ASSUMPTION: weight is 1

        value_np = value_to_numpy(neuron.getRawState().getValue())

        if check_same_facts_assumption:
            for n in layer_neurons[1:]:
                # TODO: ASSUMPTION: is not learnable
                assert not n.hasLearnableValue
                # TODO: ASSUMPTION: weight is 1

                assert (value_to_numpy(n.getRawState().getValue()) == value_np).all()

        self.value = atleast_3d_rev(torch.tensor(value_np))

    def forward(self, *kargs, **kwargs):
        return self.value


class ScalarOutput(torch.nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index

    def forward(self, layer_values: dict[int, torch.Tensor]):
        return torch.squeeze(layer_values[self.layer_index]).item()


if __name__ == "__main__":
    try:
        dataset = MyMutagenesis()
        dataset.settings.compute_neuron_layer_indices = True
        # dataset.settings.iso_value_compression = False
        # dataset.settings.chain_pruning = False
        print("Building dataset...")
        built_dataset = dataset.build(sample_run=True)

        ###### CONFIG ######

        # TODO: ASSUMPTION: all neurons have the same layer layout
        check_same_layers_assumption = False
        # TODO: ASSUMPTION: all facts have the same value
        check_same_facts_assumption = True
        # TODO: ASSUMPTION: all neurons in a given WeightedRuleLayer have the same weights
        check_same_weights_assumption = True
        # TODO: ASSUMPTION: all neurons in a given WeightedRuleLayer have the same number of inputs
        check_same_inputs_dim_assumption = True
        # TODO: ASSUMPTION: all neurons in a given WeightedAtomLayer have the same inputs
        check_same_inputs_assumption = True
        run_tests = True

        ###### DATASET CONFIG ######

        # TODO all samples at once instead
        # samples = [built_dataset.samples[107]]
        i = 108
        # i = random.choice(list(range(len(built_dataset.samples))))
        samples = [built_dataset.samples[i]]
        print("SAMPLE", i)
        # samples = built_dataset.samples

        samples[0].draw(filename="run.png", show=False)

        ###### ALGORITHM ######

        print("Layers discovery...")
        layers = discover_all_layers(samples, layers_verification=check_same_layers_assumption)

        network = get_neurons_per_layer(samples)

        ordinals_per_layer: dict[int, dict[int, int]] = {
            l: {int(n.getIndex()): o for o, n in enumerate(neurons)} for l, neurons in network.items()
        }
        ordinals: dict[int, tuple[int, int]] = {
            i: (l, o) for l, i_o_dict in ordinals_per_layer.items() for i, o in i_o_dict.items()
        }

        model = torch.nn.Sequential()

        for l in layers:
            print()
            print(f"Layer {l.index}:")
            if l.type == "FactNeuron":
                module = FactLayer(network[l.index], check_same_facts_assumption=check_same_facts_assumption)
            elif l.type == "WeightedAtomNeuron":
                module = WeightedAtomLayer(
                    network[l.index],
                    ordinals,
                    check_same_inputs_assumption=check_same_inputs_assumption,
                )
            elif l.type == "WeightedRuleNeuron":
                module = WeightedRuleLayer(
                    network[l.index],
                    ordinals,
                    check_same_weights_assumption=check_same_weights_assumption,
                    check_same_inputs_dim_assumption=check_same_inputs_dim_assumption,
                )
                if run_tests:
                    test_gather_module(module.gather, network)
            elif l.type == "AggregationNeuron":
                module = AggregationLayer(
                    network[l.index],
                    ordinals,
                    check_same_inputs_dim_assumption=check_same_inputs_dim_assumption,
                )
                if run_tests:
                    test_gather_module(module.gather, network)
            else:
                raise NotImplementedError(l.type)

            model.append(LayerPipe(module, layer_index=l.index))

        model.append(ScalarOutput(layers[-1].index))
        print(model)

        expected = value_to_numpy(samples[0].java_sample.query.neuron.getRawState().getValue())
        actual = model(None)

        print("Expected:", expected)
        print("Actual:", actual)
    except jpype.JException as e:
        print(e.message())
        print(e.stacktrace())
        raise e
