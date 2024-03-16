from collections import defaultdict, deque
from typing import Any, Protocol, Sequence
from typing import get_args as t_get_args

import jpype
import numpy as np
import torch
from neuralogic.core.builder.builder import NeuralSample
from neuralogic.core.template import Iterable
from tqdm.auto import tqdm

from lib.nn.definitions.ops import AggregationDef, TransformationDef
from lib.nn.sources.base import LayerDefinition, LayerType
from lib.other_utils import camel_to_snake


class JavaValue(Protocol):
    def getAsArray(self) -> np.ndarray: ...

    def size(self) -> Sequence[int]: ...


class JavaWeight(Protocol):
    @property
    def value(self) -> JavaValue: ...

    @property
    def index(self) -> int: ...

    def isLearnable(self) -> bool: ...


class JavaNeuron(Protocol):
    def getIndex(self) -> int: ...

    def getInputs(self) -> Sequence["JavaNeuron"]: ...

    def getRawState(self) -> Any: ...

    def getClass(self) -> Any: ...

    def getLayer(self) -> int: ...

    def getWeights(self) -> Sequence[JavaWeight]: ...

    def getOffset(self) -> JavaWeight: ...

    def getTransformation(self) -> Any: ...

    def getCombination(self) -> Any: ...


DTYPE_TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def java_value_to_numpy(java_value, dtype: torch.dtype | None = None) -> np.ndarray:
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype not in DTYPE_TORCH_TO_NUMPY:
        raise NotImplementedError(f"Conversion from {dtype} to numpy equivalent not yet implemented.")

    np_dtype = DTYPE_TORCH_TO_NUMPY[dtype]

    arr = np.asarray(java_value.getAsArray(), dtype=np_dtype)
    arr = arr.reshape(java_value.size())
    return arr


def java_value_to_tensor(java_value, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.tensor(java_value_to_numpy(java_value, dtype))


CLASS_TO_LAYER_TYPE_MAP: dict[str, LayerType] = {
    "FactNeuron": "FactLayer",
    "AtomNeuron": "AtomLayer",
    "RuleNeuron": "RuleLayer",
    "WeightedAtomNeuron": "WeightedAtomLayer",
    "WeightedRuleNeuron": "WeightedRuleLayer",
    "AggregationNeuron": "AggregationLayer",
}


def _get_layer_type(java_neuron: JavaNeuron) -> LayerType:
    class_name = str(java_neuron.getClass().getSimpleName())

    if class_name not in CLASS_TO_LAYER_TYPE_MAP:
        raise ValueError(f"Unsupported neuron class {class_name}")

    return CLASS_TO_LAYER_TYPE_MAP[class_name]


_TRANSFORMATIONS: set[TransformationDef] = set(t_get_args(TransformationDef))


def get_transformation(java_neuron: JavaNeuron) -> TransformationDef | None:
    java_transformation = java_neuron.getTransformation()

    if java_transformation is None:
        return None

    tr_class_name = str(java_transformation.getClass().getSimpleName())

    out = camel_to_snake(tr_class_name).replace("re_lu", "relu")

    if out not in _TRANSFORMATIONS:
        raise NotImplementedError(f"Unsupported transformation: {out} (Java class {tr_class_name})")

    return out


_AGGREGATIONS: set[AggregationDef] = set(t_get_args(AggregationDef))

_AggregationCls: jpype.JClass | None = None


def get_aggregation(java_neuron: JavaNeuron) -> AggregationDef | None:
    global _AggregationCls

    java_combination = java_neuron.getCombination()

    if _AggregationCls is None:
        _AggregationCls = jpype.JClass("cz.cvut.fel.ida.algebra.functions.Aggregation")

    if java_combination is None or not isinstance(java_combination, _AggregationCls):
        return None

    agg_class_name = str(java_combination.getClass().getSimpleName())

    out = camel_to_snake(agg_class_name)

    if out not in _AGGREGATIONS:
        raise NotImplementedError(f"Unsupported aggregation: {out} (Java class {agg_class_name})")

    return out


def _get_neuron(sample: NeuralSample | JavaNeuron) -> JavaNeuron:
    if isinstance(sample, NeuralSample):
        neuron = sample.java_sample.query.neuron
    else:
        neuron = sample
    return neuron


def _iter_neuron_pairs_from_neuron(neuron: JavaNeuron) -> Iterable[tuple[JavaNeuron, JavaNeuron]]:
    for n in neuron.getInputs():
        yield from _iter_neuron_pairs_from_neuron(n)
        yield n, neuron


def _iter_neuron_pairs_from_sample(sample: NeuralSample | JavaNeuron):
    yield from _iter_neuron_pairs_from_neuron(_get_neuron(sample))


def _iter_neurons(neuron: JavaNeuron) -> Iterable[JavaNeuron]:
    yield neuron
    for n in neuron.getInputs():
        yield from _iter_neurons(n)


def _iter_neurons_from_sample(sample: NeuralSample | JavaNeuron):
    yield from _iter_neurons(_get_neuron(sample))


def _discover_layers_from_sample(sample: NeuralSample | JavaNeuron) -> list[LayerDefinition]:
    layer_id_pairs = list(((int(a.getLayer()), int(b.getLayer())) for a, b in _iter_neuron_pairs_from_sample(sample)))
    arr = np.array(layer_id_pairs)
    arr = np.unique(arr, axis=0)
    uniq = np.unique(arr).tolist()

    predecessors: dict[int, int] = {}
    for k in uniq:
        mask = arr[:, 1] == k
        if np.any(mask):
            predecessors[k] = int(np.min(arr[mask][:, 0]))

    types: dict[int, LayerType] = {}
    for n in _iter_neurons_from_sample(sample):
        l = int(n.getLayer())
        if l not in types:
            types[l] = _get_layer_type(n)

        if len(types) == len(uniq):
            break

    order = [int(_get_neuron(sample).getLayer())]
    while len(order) < len(uniq):
        order.append(predecessors[order[-1]])
    order.reverse()

    out = [LayerDefinition(id=id, type=types[id]) for id in order]
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
