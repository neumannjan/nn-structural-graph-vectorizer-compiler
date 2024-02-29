from collections import defaultdict, deque
from typing import Any, Protocol, Sequence

import numpy as np
from neuralogic.core.builder.builder import NeuralSample
from tqdm.auto import tqdm

from lib.nn.sources.source import LayerDefinition, LayerDefinitions, LayerOrdinal, LayerType
from lib.nn.topological.settings import Settings


class JavaValue(Protocol):
    def getAsArray(self) -> np.ndarray:
        ...

    def size(self) -> Sequence[int]:
        ...


class JavaWeight(Protocol):
    @property
    def value(self) -> JavaValue:
        ...

    @property
    def index(self) -> int:
        ...

    def isLearnable(self) -> bool:
        ...


class JavaNeuron(Protocol):
    def getIndex(self) -> int:
        ...

    def getInputs(self) -> Sequence["JavaNeuron"]:
        ...

    def getRawState(self) -> Any:
        ...

    def getClass(self) -> Any:
        ...

    def getLayer(self) -> int:
        ...

    def getWeights(self) -> Sequence[JavaWeight]:
        ...

    def getOffset(self) -> JavaWeight:
        ...


CLASS_TO_LAYER_TYPE_MAP: dict[str, LayerType] = {
    "FactNeuron": "FactLayer",
    "WeightedAtomNeuron": "WeightedAtomLayer",
    "WeightedRuleNeuron": "WeightedRuleLayer",
    "AggregationNeuron": "AggregationLayer",
    "RuleNeuron": "RuleLayer"
}


def _get_layer_type(java_neuron: JavaNeuron) -> LayerType:
    class_name = str(java_neuron.getClass().getSimpleName())

    if class_name not in CLASS_TO_LAYER_TYPE_MAP:
        raise ValueError(f"Unsupported neuron class {class_name}")

    return CLASS_TO_LAYER_TYPE_MAP[class_name]


def _discover_layers_from_sample(sample: NeuralSample | JavaNeuron) -> list[LayerDefinition]:
    out: list[LayerDefinition] = []

    if isinstance(sample, NeuralSample):
        neuron = sample.java_sample.query.neuron
    else:
        neuron = sample
    initial_layer: int = neuron.getLayer()
    neighbor_layers: dict[int, int] = {}
    layer_types: dict[int, LayerType] = {}

    queue = deque([neuron])

    while len(queue) > 0:
        neuron = queue.popleft()
        layer = neuron.getLayer()
        layer_type = _get_layer_type(neuron)
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
    out += [LayerDefinition(layer, layer_types[layer])]
    while True:
        if layer not in neighbor_layers:
            break

        layer = neighbor_layers[layer]
        out += [LayerDefinition(layer, layer_types[layer])]

    out.reverse()
    return out


def discover_layers(
    samples: Sequence[NeuralSample | JavaNeuron], check_same_layers_assumption: bool
) -> tuple[LayerDefinition, ...]:
    layers = tuple(_discover_layers_from_sample(samples[0]))

    if check_same_layers_assumption:
        for sample in tqdm(samples[1:], desc="Verifying layers"):
            assert tuple(_discover_layers_from_sample(sample)) == layers

    return layers


def compute_java_neurons_per_layer(samples: Sequence[NeuralSample | JavaNeuron]) -> dict[int, list[JavaNeuron]]:
    queue = deque(
        (sample.java_sample.query.neuron if isinstance(sample, NeuralSample) else sample for sample in samples)
    )

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


def compute_java_ordinals_for_layer(
    layer_def: LayerDefinition, neurons: list[JavaNeuron], settings: Settings
) -> dict[int, LayerOrdinal]:
    if settings.assume_facts_same and layer_def.type == "FactLayer":
        return {int(n.getIndex()): LayerOrdinal(layer_def.id, 0) for n in neurons}

    return {int(n.getIndex()): LayerOrdinal(layer_def.id, i) for i, n in enumerate(neurons)}


def compute_java_neurons(
    neurons_per_layer: dict[int, list[JavaNeuron]], layer_definitions: LayerDefinitions
) -> dict[int, JavaNeuron]:
    out: dict[int, JavaNeuron] = {}

    for layer in layer_definitions:
        for java_neuron in neurons_per_layer[layer.id]:
            out[java_neuron.getIndex()] = java_neuron

    return out