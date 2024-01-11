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
        layer_type = str(neuron.getClass().getSimpleName())
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

        self.per_layer_ordinals = {l: torch.tensor(ordinals) for l, ordinals in per_layer_ordinals.items()}
        self.concatenated_ordinals = torch.tensor(concatenated_ordinals)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        layer_inputs_needed = torch.concatenate(
            [torch.gather(layer_values[layer], 0, self.per_layer_ordinals[layer]) for layer in self.layers]
        )

        out = torch.gather(layer_inputs_needed, 0, self.concatenated_ordinals)
        return out


def test_gather_module(gather_module: GatherModule, network: dict[int, list]):
    # input: indices of the neurons (so that for each neuron, its index is in its position)
    layer_values = {l: torch.tensor([n.getIndex() for n in neurons]) for l, neurons in network.items()}

    # expected output: list of the neuron indices that the module is supposed to gather
    expected = torch.tensor([network[l][o].getIndex() for l, o in gather_module.input_layer_ordinal_pairs])

    # actual output:
    actual = gather_module(layer_values)

    # assert
    assert (actual == expected).all()


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self, layer_neurons: list, neuron_ordinals: dict[int, tuple[int, int]], check_same_weights_assumption=True
    ) -> None:
        super().__init__()

        self.gather_module = GatherModule(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()]
        )

        neuron = layer_neurons[0]

        self.layer_index = neuron.getLayer()
        self.weights_map = torch.nn.ParameterDict()
        self.all_weights = []

        self.weight_indices = []
        for weight in tqdm(neuron.getWeights(), desc="Weights"):
            self.weight_indices += [weight.index]

            w_npy = np.asarray(weight.value.getAsArray())
            w_npy = w_npy.reshape(weight.value.size())
            w_tensor = torch.tensor(w_npy)

            if weight.isLearnable:
                if int(weight.index) in self.weights_map:
                    w_tensor = self.weights_map[str(weight.index)]
                else:
                    self.weights_map[str(weight.index)] = w_tensor = torch.nn.Parameter(w_tensor)
            self.all_weights += [w_tensor]

        if check_same_weights_assumption:
            for neuron in tqdm(layer_neurons[1:], desc="Verifying neurons"):
                assert neuron.getLayer() == self.layer_index

                for our_widx, weight in zip(self.weight_indices, neuron.getWeights()):
                    assert our_widx == weight.index


class WeightedAtomLayer(torch.nn.Module):
    def __init__(
        self, layer_neurons: list, neuron_ordinals: dict[int, tuple[int, int]], check_same_inputs_assumption=True
    ) -> None:
        super().__init__()

        neuron = layer_neurons[0]

        # TODO: ASSUMPTION: all inputs are from the same layer
        # (easy fix: replace with GatherModule)
        input_layer_ordinal_pairs = [neuron_ordinals[int(n.getIndex())] for n in neuron.getInputs()]
        input_layers = set((l for l, _ in input_layer_ordinal_pairs))

        assert len(input_layers) == 1
        self.input_layer = next(iter(input_layers))
        self.input_ordinals = [o for _, o in input_layer_ordinal_pairs]

        if check_same_inputs_assumption:
            first_inputs = tuple((int(inp.getIndex()) for inp in neuron.getInputs()))
            print(first_inputs)
            for n in layer_neurons[1:]:
                this_inputs = tuple((int(inp.getIndex()) for inp in n.getInputs()))
                print(this_inputs)
                assert first_inputs == this_inputs

        ###### WEIGHTS
        self.weights_map = torch.nn.ParameterDict()
        self.all_weights = []

        self.weight_indices = []
        for neuron, weight in tqdm([(n, w) for n in layer_neurons for w in neuron.getWeights()], desc="Weights"):
            self.weight_indices += [weight.index]

            w_npy = np.asarray(weight.value.getAsArray())
            w_npy = w_npy.reshape(weight.value.size())
            w_tensor = torch.tensor(w_npy)

            if weight.isLearnable:
                if int(weight.index) in self.weights_map:
                    w_tensor = self.weights_map[str(weight.index)]
                else:
                    self.weights_map[str(weight.index)] = w_tensor = torch.nn.Parameter(w_tensor)
            self.all_weights += [w_tensor]


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
        # TODO: ASSUMPTION: all neurons in a given WeightedRuleLayer have the same weights
        check_same_weights_assumption = True
        # TODO: ASSUMPTION: all neurons in a given WeightedAtomLayer have the same inputs
        check_same_inputs_assumption = True
        run_tests = True

        ###### DATASET CONFIG ######

        # TODO all samples at once instead
        samples = [built_dataset.samples[108]]
        # samples = built_dataset.samples

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
            if l.type == "WeightedAtomNeuron":
                module = WeightedAtomLayer(
                    network[l.index],
                    ordinals,
                    check_same_inputs_assumption=check_same_inputs_assumption,
                )
                model.append(module)
            elif l.type == "WeightedRuleNeuron":
                module = WeightedRuleLayer(
                    network[l.index],
                    ordinals,
                    check_same_weights_assumption=check_same_weights_assumption,
                )
                model.append(module)

                if run_tests:
                    test_gather_module(module.gather_module, network)

        print(model)
    except jpype.JException as e:
        print(e.message())
        print(e.stacktrace())
        raise e
