from collections import defaultdict, deque
from typing import Any, Protocol, Sequence
from typing import get_args as t_get_args

import jpype
import numpy as np
import torch
from neuralogic.core.builder.builder import NeuralSample
from neuralogic.core.template import Iterable

from lib.facts.model import get_rule_or_fact_main_name
from lib.facts.parser import ParserError, parse_rule_or_fact
from lib.model.ops import AggregationDef, TransformationDef
from lib.sources.base import LayerDefinition, LayerType
from lib.utils import camel_to_snake


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

    @property
    def name(self) -> str: ...


DTYPE_TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def java_value_to_numpy(java_value, dtype: torch.dtype) -> np.ndarray:
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype not in DTYPE_TORCH_TO_NUMPY:
        raise NotImplementedError(f"Conversion from {dtype} to numpy equivalent not yet implemented.")

    np_dtype = DTYPE_TORCH_TO_NUMPY[dtype]

    arr = np.asarray(java_value.getAsArray(), dtype=np_dtype)
    arr = arr.reshape(java_value.size())
    return arr


def java_value_to_tensor(java_value, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(java_value_to_numpy(java_value, dtype), dtype=dtype)


CLASS_TO_LAYER_TYPE_MAP: dict[str, LayerType] = {
    "FactNeuron": "FactLayer",
    "AtomNeuron": "AtomLayer",
    "RuleNeuron": "RuleLayer",
    "WeightedAtomNeuron": "WeightedAtomLayer",
    "WeightedRuleNeuron": "WeightedRuleLayer",
    "AggregationNeuron": "AggregationLayer",
}


_LAYER_TYPE_TO_ABBREV: dict[LayerType, str] = {
    "FactLayer": "f",
    "AtomLayer": "a",
    "RuleLayer": "r",
    "WeightedRuleLayer": "wr",
    "WeightedAtomLayer": "wa",
    "AggregationLayer": "ag",
}


def _get_layer_type(java_neuron: JavaNeuron) -> LayerType:
    class_name = str(java_neuron.getClass().getSimpleName())

    if class_name not in CLASS_TO_LAYER_TYPE_MAP:
        raise ValueError(f"Unsupported neuron class {class_name}")

    return CLASS_TO_LAYER_TYPE_MAP[class_name]


_TRANSFORMATIONS: set[TransformationDef] = set(t_get_args(TransformationDef))


def get_transformation(java_neuron: JavaNeuron) -> TransformationDef:
    java_transformation = java_neuron.getTransformation()

    if java_transformation is None:
        return "identity"

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


def _iter_neuron_pairs_from_neuron(neuron: JavaNeuron) -> Iterable[tuple[JavaNeuron, JavaNeuron]]:
    for n in neuron.getInputs():
        yield from _iter_neuron_pairs_from_neuron(n)
        yield n, neuron


def _iter_neurons(neuron: JavaNeuron) -> Iterable[JavaNeuron]:
    yield neuron
    for n in neuron.getInputs():
        yield from _iter_neurons(n)


_Key = tuple[int, str, LayerType]


def _layer_order_key(key: _Key):
    layer_ord, _, layer_type = key

    a = 0 if layer_type == "FactLayer" else 1
    b = -layer_ord

    return a, b


def compute_java_neurons_per_layer(
    samples: Sequence[NeuralSample | JavaNeuron],
) -> tuple[dict[str, list[JavaNeuron]], list[LayerDefinition]]:
    queue = deque(
        (sample.java_sample.query.neuron if isinstance(sample, NeuralSample) else sample for sample in samples)
    )

    visited = set()

    neurons_per_layer: dict[_Key, list[JavaNeuron]] = defaultdict(lambda: [])

    names = set()

    try:
        while len(queue) > 0:
            neuron = queue.popleft()

            neuron_index = int(neuron.getIndex())
            if neuron_index in visited:
                continue

            visited.add(neuron_index)
            neuron_rule_name = get_rule_or_fact_main_name(parse_rule_or_fact(str(neuron.name)))
            names.add(neuron_rule_name)
            neurons_per_layer[int(neuron.getLayer()), neuron_rule_name, _get_layer_type(neuron)].append(neuron)

            for inp in neuron.getInputs():
                inp_index = int(inp.getIndex())
                if inp_index not in visited:
                    queue.append(inp)
    except ParserError as e:
        raise ValueError(f"Failed to parse rule name from '{str(neuron.name)}'") from e

    layers = sorted(neurons_per_layer.keys(), key=_layer_order_key)

    out: dict[str, list] = {}
    layer_defs: list[LayerDefinition] = []

    for k in layers:
        _, n, t = k

        id: str = n + "__" + _LAYER_TYPE_TO_ABBREV[t]

        out[id] = neurons_per_layer[k]
        layer_defs.append(LayerDefinition(id=id, type=t))

    return out, layer_defs
