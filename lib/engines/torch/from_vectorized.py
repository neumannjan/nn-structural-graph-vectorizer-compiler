from lib.engines.torch.dim_reduce import build_dim_reduce_module
from lib.engines.torch.gather import GenericGatherModule, RepeatModule, SliceValuesModule, TakeValueModule
from lib.engines.torch.linear import LinearModule
from lib.engines.torch.network import LayerModule, NetworkModule, NetworkParams
from lib.engines.torch.refs import ConcatRefsModule, RetrieveRefModule
from lib.engines.torch.scatter import build_scatter_module, counts_to_index
from lib.engines.torch.transform import build_transformation
from lib.engines.torch.view import ViewModule
from lib.vectorize.model import *

import torch


def _get_fact_value(fact: Fact, shape: ConcreteShape) -> torch.Tensor:
    match fact:
        case UnitFact():
            return torch.ones(shape.dims).unsqueeze(0)
        case EyeFact(dim=dim):
            return torch.eye(dim).unsqueeze(0)
        case ValueFact(value=value):
            return torch.tensor(value)
        case _:
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

        value = torch.tensor(weight_def.value)
        params[key] = torch.nn.Parameter(value, requires_grad=True)

    return NetworkParams(params)


def _get_refs(refs: LayerRefs) -> list[str]:
    out = []
    out.extend(refs.facts)
    out.extend(refs.weights)
    out.extend(refs.layers)
    return out


def _for_op(op: Operation, allow_non_builtin_torch_ops: bool) -> torch.nn.Module:
    match op:
        case LayerRefs():
            refs = _get_refs(op)
            if len(refs) == 1:
                return RetrieveRefModule(refs[0])
            else:
                return ConcatRefsModule(refs)
        case GatherPair(a, b):
            first_gathers = _for_op(a, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops)
            last_gather = _for_op(b, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops)

            if isinstance(first_gathers, torch.nn.Sequential):
                first_gathers.append(last_gather)
                return first_gathers
            else:
                return torch.nn.Sequential(first_gathers, last_gather)
        case Linear(weight_refs=weight_refs, additional_ops=additional_ops):
            weight_module = torch.nn.Sequential()

            retrieve_refs = _for_op(
                weight_refs,
                allow_non_builtin_torch_ops=allow_non_builtin_torch_ops,
            )

            weight_module.append(retrieve_refs)

            for op in additional_ops.operations:
                op_module = _for_op(op, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops)
                weight_module.append(op_module)

            return LinearModule(weight_module)
        case GenericGather(ordinals=ordinals):
            return GenericGatherModule(ordinals)
        case TakeSingleValue(ordinal=ordinal):
            return TakeValueModule(ordinal)
        case SliceValues(start=start, end=end, step=step):
            return SliceValuesModule(start, end, step)
        case Repeat(times=times, total_length=total_length):
            return RepeatModule(repeats=times, total_length=total_length)
        case Transform(transform=transform):
            return build_transformation(transform)
        case DimReduce(dim=dim, reduce=reduce):
            return build_dim_reduce_module(dim, reduce)
        case UnevenReduce(counts=counts, reduce=reduce):
            index = counts_to_index(counts)
            return build_scatter_module(
                index=index,
                reduce=reduce,
                allow_non_builtin_torch_ops=allow_non_builtin_torch_ops,
            )
        case View(shape=shape):
            return ViewModule(shape.dims)
        case _:
            assert False, f"{op}"


def _for_batch(
    batch_reference: OpSeqBatch,
    debug: bool,
    allow_non_builtin_torch_ops: bool,
) -> torch.nn.Module:
    modules: list[torch.nn.Module] = []

    for key, layer in batch_reference.layers.items():
        layer_modules: list[torch.nn.Module] = [
            _for_op(op, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops) for op in layer.operations
        ]

        layer_module = LayerModule(layer_modules, out_key=key, debug=debug)
        modules.append(layer_module)

    assert isinstance(key, str)
    # retrieve last layer ref at batch output
    modules.append(RetrieveRefModule(key))

    return torch.nn.Sequential(*modules)


def build_torch_network(
    reference: VectorizedOpSeqNetwork,
    debug: bool,
    allow_non_builtin_torch_ops: bool,
) -> torch.nn.Module:
    params_module = _build_params_module(reference)

    batch_modules = torch.nn.ModuleList()

    for i, (batch_id, batch_ref) in enumerate(reference.batches.items()):
        assert i == batch_id
        batch_module = _for_batch(
            batch_ref,
            debug=debug,
            allow_non_builtin_torch_ops=allow_non_builtin_torch_ops,
        )

        batch_modules.append(batch_module)

    return NetworkModule(params_module=params_module, batch_modules=batch_modules)