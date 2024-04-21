from typing import Iterable, Mapping

from lib.vectorize.model import *

_Node = tuple[int, str]


class _ChainGraph:
    __slots__ = ("nodes", "single_successors", "single_predecessors", "ref_gathers")

    def __init__(
        self,
        single_successors: Mapping[_Node, _Node],
        single_predecessors: Mapping[_Node, _Node],
        ref_gathers: Mapping[_Node, GatheredLayers | None],
    ) -> None:
        self.single_successors = single_successors
        self.single_predecessors = single_predecessors
        self.ref_gathers = ref_gathers

    def iter_beginnings(self) -> Iterable[_Node]:
        for n in self.single_successors:
            if n not in self.single_predecessors:
                yield n

    def iter_single_chain_with_ref_gathers(self, beginning: _Node) -> Iterable[tuple[_Node, GatheredLayers | None]]:
        n = beginning
        gather = self.ref_gathers.get(n, None)
        yield n, gather

        while True:
            try:
                n = self.single_successors[n]
                gather = self.ref_gathers.get(n, None)
                yield n, gather
            except KeyError:
                return

    def iter_single_chain(self, beginning: _Node) -> Iterable[_Node]:
        n = beginning
        yield n

        while True:
            try:
                n = self.single_successors[n]
                yield n
            except KeyError:
                return

    def iter_chains_with_ref_gathers(self) -> Iterable[Iterable[tuple[_Node, GatheredLayers | None]]]:
        for n in self.iter_beginnings():
            yield self.iter_single_chain_with_ref_gathers(beginning=n)

    def iter_chains(self) -> Iterable[Iterable[_Node]]:
        for n in self.iter_beginnings():
            yield self.iter_single_chain(beginning=n)

    def iter_chain_lengths(self) -> Iterable[int]:
        for chain in self.iter_chains():
            yield sum((1 for _ in chain))


class ComputeChainGraph:
    def __init__(self, network: VectorizedOpSeqNetwork | VectorizedLayerNetwork, types: tuple[int, ...]) -> None:
        self.network = network
        self.searched_types = types

        if LayerRefs.TYPE_LAYER not in types:
            raise ValueError()

    def _iter_refs(self, refs: LayerRefs | Refs) -> Iterable[_Node]:
        for t, l in zip(refs.types, refs.layer_ids):
            if t in self.searched_types:
                yield t, l

    def _iter_refs_for_ops(self, ops: OperationSeq) -> Iterable[_Node]:
        if ops.layer_refs is not None:
            yield from self._iter_refs(ops.layer_refs)

        for op in ops.operations:
            match op:
                case Linear(weight_ops=OperationSeq(layer_refs=LayerRefs() as refs)):
                    yield from self._iter_refs(refs)

    def _iter_refs_for_input(self, input: Input) -> Iterable[_Node]:
        match input:
            case GatheredLayers(refs=refs):
                yield from self._iter_refs(refs)
            case Refs():
                yield from self._iter_refs(input)
            case _:
                assert False, f"{input}"

    def _iter_all_refs(self, layers: Mapping[str, OperationSeq | Layer]) -> Iterable[_Node]:
        for ops in layers.values():
            match ops:
                case OperationSeq():
                    yield from self._iter_refs_for_ops(ops)
                case Layer(base=InputLayerBase(input=input)):
                    yield from self._iter_refs_for_input(input)
                case Layer(base=LinearLayerBase(input=input, weight=weight)):
                    yield from self._iter_refs_for_input(input)
                    yield from self._iter_refs_for_input(weight)
                case Layer(base=LinearGatherLayerBase(input=input, weight=weight)):
                    yield from self._iter_refs_for_input(input)
                    yield from self._iter_refs_for_input(weight)

    def __call__(self, layers: Mapping[str, OperationSeq | Layer]) -> _ChainGraph:
        succ: dict[_Node, _Node] = {}
        ref_gathers: dict[_Node, GatheredLayers | None] = {}

        for layer_id, ops in layers.items():
            match ops:
                case OperationSeq(
                    layer_refs=LayerRefs(types=[t], layer_ids=[ref_layer_id])
                ) if t in self.searched_types:
                    succ[t, ref_layer_id] = LayerRefs.TYPE_LAYER, layer_id
                case OperationSeq(operations=operations):
                    for op in operations:
                        match op:
                            case Linear(
                                weight_ops=OperationSeq(layer_refs=LayerRefs(types=[t], layer_ids=[ref_layer_id]))
                            ) if t in self.searched_types:
                                succ[t, ref_layer_id] = LayerRefs.TYPE_LAYER, layer_id
                                break
                case Layer(
                    base=(
                        InputLayerBase(input=Refs())
                        | LinearLayerBase(input=Refs())
                        | LinearLayerBase(weight=Refs())
                        | LinearGatherLayerBase(input=Refs())
                        | LinearGatherLayerBase(weight=Refs())
                    )
                ):
                    raise NotImplementedError()
                case Layer(
                    base=(
                        InputLayerBase(input=GatheredLayers(refs=LayerRefs(types=[t], layer_ids=[ref_layer_id])) as gl)
                        | LinearLayerBase(
                            input=GatheredLayers(refs=LayerRefs(types=[t], layer_ids=[ref_layer_id])) as gl
                        )
                        | LinearLayerBase(
                            weight=GatheredLayers(refs=LayerRefs(types=[t], layer_ids=[ref_layer_id])) as gl
                        )
                        | LinearGatherLayerBase(
                            input=GatheredLayers(refs=LayerRefs(types=[t], layer_ids=[ref_layer_id])) as gl
                        )
                        | LinearGatherLayerBase(
                            weight=GatheredLayers(refs=LayerRefs(types=[t], layer_ids=[ref_layer_id])) as gl
                        )
                    )
                ) if t in self.searched_types:
                    succ[t, ref_layer_id] = LayerRefs.TYPE_LAYER, layer_id
                    if (t, ref_layer_id) in ref_gathers:
                        ref_gathers[t, ref_layer_id] = None
                    else:
                        ref_gathers[t, ref_layer_id] = gl

        visited: set[_Node] = set()

        for ref in self._iter_all_refs(layers):
            if ref not in visited:
                visited.add(ref)
            else:
                succ.pop(ref, None)

        pred: dict[_Node, _Node] = {b: a for a, b in succ.items()}

        return _ChainGraph(succ, pred, ref_gathers)
