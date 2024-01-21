from collections import defaultdict, deque
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from lib.interfaces import JavaNeuron
from lib.nn.topological.settings import Settings
from neuralogic.core.builder.builder import NeuralSample
from tqdm.auto import tqdm


@dataclass(frozen=True)
class LayerDefinition:
    type: str
    index: int


def get_neuron_type(java_neuron: JavaNeuron) -> str:
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
            assert (
                layer_types[layer] == layer_type
            ), f"Layer {layer} found of types {layer_types[layer]} and {layer_type}"
        else:
            layer_types[layer] = layer_type

        layer_inputs = list(neuron.getInputs())

        if len(layer_inputs) == 0:
            break

        for inp in layer_inputs:
            if inp.getLayer() == layer:
                raise RuntimeError(
                    f"Neuron in layer {layer} has an input in layer {inp.getLayer()}. This should not happen. "
                    f"Did you do the sample run?"
                )

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


def discover_all_layers(samples: Sequence[NeuralSample], settings: Settings) -> tuple[LayerDefinition, ...]:
    layers = tuple(discover_layers(samples[0]))

    if settings.check_same_layers_assumption:
        for sample in tqdm(samples[1:], desc="Verifying layers"):
            assert tuple(discover_layers(sample)) == layers

    return layers


TopologicalNetwork = dict[int, list[JavaNeuron]]


def get_neurons_per_layer(samples: Sequence[NeuralSample]) -> TopologicalNetwork:
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


class LayerOrdinal(NamedTuple):
    layer: int
    ordinal: int


Ordinals = dict[int, LayerOrdinal]
OrdinalsPerLayer = dict[int, Ordinals]


def compute_neuron_ordinals_for_layer(
    layer_def: LayerDefinition, neurons: list, settings: Settings
) -> dict[int, LayerOrdinal]:
    if settings.assume_facts_same and layer_def.type == "FactNeuron":
        return {n.getIndex(): LayerOrdinal(layer_def.index, 0) for n in neurons}

    return {n.getIndex(): LayerOrdinal(layer_def.index, i) for i, n in enumerate(neurons)}


def compute_neuron_ordinals(
    layers: Sequence[LayerDefinition], network: TopologicalNetwork, settings: Settings
) -> tuple[OrdinalsPerLayer, Ordinals]:
    ordinals_per_layer: OrdinalsPerLayer = {
        l.index: compute_neuron_ordinals_for_layer(l, network[l.index], settings) for l in layers
    }
    ordinals: Ordinals = {i: o for _, i_o_dict in ordinals_per_layer.items() for i, o in i_o_dict.items()}

    return ordinals_per_layer, ordinals
