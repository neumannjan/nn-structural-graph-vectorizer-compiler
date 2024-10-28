from typing import OrderedDict

import numpy as np
import torch

from compute_graph_vectorize.engines.torch.dim_reduce import build_dim_reduce_module
from compute_graph_vectorize.engines.torch.gather import (
    GenericGatherModule,
    RepeatInterleaveModule,
    RepeatModule,
    SliceValuesModule,
    TakeValueModule,
)
from compute_graph_vectorize.engines.torch.linear import LinearModule
from compute_graph_vectorize.engines.torch.network import LayerModule, NetworkModule, NetworkParams
from compute_graph_vectorize.engines.torch.refs import ConcatRefsModule, RetrieveRefModule
from compute_graph_vectorize.engines.torch.scatter import build_scatter_module, counts_to_index
from compute_graph_vectorize.engines.torch.settings import TORCH_SETTINGS_DEFAULT, TorchModuleSettings
from compute_graph_vectorize.engines.torch.transform import build_transformation
from compute_graph_vectorize.engines.torch.view import ViewModule
from compute_graph_vectorize.vectorize.model import *


def _get_fact_value(fact: Fact, shape: ConcreteShape) -> torch.Tensor:
    if isinstance(fact, UnitFact):
        return torch.ones(shape.dims, dtype=torch.get_default_dtype()).unsqueeze(0)
    if isinstance(fact, EyeFact):
        return torch.eye(fact.dim, dtype=torch.get_default_dtype()).unsqueeze(0)
    if isinstance(fact, ValueFact):
        return torch.tensor(fact.value, dtype=torch.get_default_dtype())
    else:
        assert False, f"{fact}"


def _build_params_module(reference: VectorizedOpSeqNetwork) -> NetworkParams:
    params = torch.nn.ParameterDict()

    for key, fact_def in reference.fact_layers.items():
        assert isinstance(fact_def.shape, ConcreteShape)
        assert key not in params

        value = torch.concat([_get_fact_value(fact, fact_def.shape) for fact in fact_def.facts], dim=0)
        params[key] = torch.nn.Parameter(value, requires_grad=False)

    for key, weight_def in reference.weights.items():
        assert key not in params

        value = torch.tensor(weight_def.value, dtype=torch.get_default_dtype())
        params[key] = torch.nn.Parameter(value, requires_grad=True)

    return NetworkParams(params)


def _for_op(op: Operation | LayerRefs, settings: TorchModuleSettings) -> torch.nn.Module:
    if isinstance(op, LayerRefs):
        refs = op.layer_ids
        if len(refs) == 1:
            return RetrieveRefModule(refs[0])
        else:
            return ConcatRefsModule(refs)
    elif isinstance(op, GatherPair):
        first_gathers = _for_op(op.a, settings)
        last_gather = _for_op(op.b, settings)

        if isinstance(first_gathers, torch.nn.Sequential):
            first_gathers.append(last_gather)
            return first_gathers
        else:
            return torch.nn.Sequential(first_gathers, last_gather)
    elif isinstance(op, Linear):
        weight_module = torch.nn.Sequential()

        if op.weight_ops.layer_refs is not None:
            retrieve_refs = _for_op(op.weight_ops.layer_refs, settings)

            weight_module.append(retrieve_refs)

        for op in op.weight_ops.operations:
            op_module = _for_op(op, settings)
            weight_module.append(op_module)

        return LinearModule(weight_module)
    elif isinstance(op, GenericGather):
        return GenericGatherModule(op.ordinals)
    elif isinstance(op, TakeSingleValue):
        return TakeValueModule(op.ordinal)
    elif isinstance(op, SliceValues):
        return SliceValuesModule(op.start, op.end, op.step)
    elif isinstance(op, Repeat):
        return RepeatModule(repeats=op.times, total_length=op.total_length)
    elif isinstance(op, RepeatInterleave):
        return RepeatInterleaveModule(repeats=op.times, total_length=op.total_length)
    elif isinstance(op, Transform):
        return build_transformation(op.transform)
    elif isinstance(op, DimReduce):
        return build_dim_reduce_module(op.dim, op.reduce)
    elif isinstance(op, UnevenReduce):
        index = counts_to_index(op.counts)
        return build_scatter_module(
            index=index,
            counts=op.counts,
            reduce=op.reduce,
            reduce_method=settings.reduce_method,
        )
    elif isinstance(op, View):
        return ViewModule(op.shape.dims)
    else:
        assert False, f"{op}"


def _for_batch(
    batch_reference: OpSeqBatch,
    debug: bool,
    settings: TorchModuleSettings,
    final_layer_only: bool = True,
) -> torch.nn.Module:
    modules: list[torch.nn.Module] = []

    for key, layer in batch_reference.layers.items():
        layer_modules: list[torch.nn.Module] = []

        if layer.layer_refs is not None:
            layer_modules.append(_for_op(layer.layer_refs, settings))

        for op in layer.operations:
            layer_modules.append(_for_op(op, settings))

        layer_module = LayerModule(layer_modules, out_key=key, expected_count=layer.expected_count, debug=debug)
        modules.append(layer_module)

    assert isinstance(key, str)

    if final_layer_only:
        # retrieve last layer ref at batch output
        modules.append(RetrieveRefModule(key))

    if len(modules) == 0:
        return modules[0]

    return torch.nn.Sequential(*modules)


def build_torch_model(
    reference: VectorizedOpSeqNetwork,
    settings: TorchModuleSettings,
    debug: bool,
    final_layer_only: bool = True,
) -> NetworkModule:
    params_module = _build_params_module(reference)

    batch_modules = torch.nn.ModuleList()

    for i, (batch_id, batch_ref) in enumerate(reference.batches.items()):
        assert i == batch_id
        batch_module = _for_batch(
            batch_ref,
            debug=debug,
            settings=settings,
            final_layer_only=final_layer_only,
        )
        batch_modules.append(batch_module)

    out = NetworkModule(params_module=params_module, batch_modules=batch_modules)

    if settings.compilation == "trace":
        out = torch.jit.trace_module(out, {"forward": ()}, strict=False)
    elif settings.compilation == "script":
        out = torch.jit.script(out)

    return out  # pyright: ignore


def torch_simple_forward_pass_runner(network: VectorizedOpSeqNetwork):
    tnetwork = build_torch_model(network, debug=False, settings=TORCH_SETTINGS_DEFAULT, final_layer_only=False)

    out: dict[int, dict[str, np.ndarray]] = {}

    for batch_id in network.batches:
        with torch.no_grad():
            batch_out: dict[str, torch.Tensor] = tnetwork()

        batch_out_np: OrderedDict[str, np.ndarray] = OrderedDict(
            ((k, v.detach().cpu().numpy()) for k, v in batch_out.items())
        )

        out[batch_id] = batch_out_np

    return out


def extract_weights_from_torch_model(module: NetworkModule) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    params = module.params_module.params
    for key in params:
        if key.startswith("w_"):
            out[key[2:]] = params[key].detach().cpu()

    return out
