from typing import Collection, Iterable, Mapping

from lib.vectorize.model import *


class _ChainGraph:
    __slots__ = ("nodes", "single_successors", "single_predecessors", "ref_gathers")

    def __init__(
        self,
        nodes: Collection[str],
        single_successors: Mapping[str, str],
        single_predecessors: Mapping[str, str],
        ref_gathers: Mapping[str, GatheredLayers],
    ) -> None:
        self.nodes = nodes
        self.single_successors = single_successors
        self.single_predecessors = single_predecessors
        self.ref_gathers = ref_gathers

    def iter_beginnings(self, g: "_ChainGraph") -> Iterable[str]:
        for n in g.nodes:
            if n not in g.single_predecessors:
                yield n

    def iter_single_chain_with_ref_gathers(
        self, g: "_ChainGraph", beginning: str
    ) -> Iterable[tuple[str, GatheredLayers | None]]:
        n = beginning
        yield n, None

        while True:
            try:
                n = g.single_successors[n]
                gather = g.ref_gathers.get(n, None)
                yield n, gather
            except KeyError:
                return

    def iter_single_chain(self, g: "_ChainGraph", beginning: str) -> Iterable[str]:
        n = beginning
        yield n

        while True:
            try:
                n = g.single_successors[n]
                yield n
            except KeyError:
                return

    def iter_chains_with_ref_gathers(self, g: "_ChainGraph") -> Iterable[Iterable[tuple[str, GatheredLayers | None]]]:
        for n in self.iter_beginnings(g):
            yield self.iter_single_chain_with_ref_gathers(g, beginning=n)

    def iter_chains(self, g: "_ChainGraph") -> Iterable[Iterable[str]]:
        for n in self.iter_beginnings(g):
            yield self.iter_single_chain(g, beginning=n)


class ComputeChainGraph:
    def __init__(self, network: VectorizedOpSeqNetwork | VectorizedLayerNetwork) -> None:
        self.network = network

    def _iter_layers_in_refs(self, refs: LayerRefs | Refs) -> Iterable[str]:
        for t, l in zip(refs.types, refs.layer_ids):
            if t == LayerRefs.TYPE_LAYER:
                yield l

    def _iter_layer_refs_for_ops(self, ops: OperationSeq) -> Iterable[str]:
        if ops.layer_refs is not None:
            yield from self._iter_layers_in_refs(ops.layer_refs)

        for op in ops.operations:
            match op:
                case Linear(weight_ops=OperationSeq(layer_refs=LayerRefs() as refs)):
                    yield from self._iter_layers_in_refs(refs)

    def _iter_layer_refs_for_input(self, input: Input) -> Iterable[str]:
        match input:
            case GatheredLayers(refs=refs):
                yield from self._iter_layers_in_refs(refs)
            case Refs():
                yield from self._iter_layers_in_refs(input)
            case _:
                assert False, f"{input}"

    def _iter_all_layer_refs(self, layers: Mapping[str, OperationSeq | Layer]) -> Iterable[str]:
        for ops in layers.values():
            match ops:
                case OperationSeq():
                    yield from self._iter_layer_refs_for_ops(ops)
                case Layer(base=InputLayerBase(input=input)):
                    yield from self._iter_layer_refs_for_input(input)
                case Layer(base=LinearLayerBase(input=input, weight=weight)):
                    yield from self._iter_layer_refs_for_input(input)
                    yield from self._iter_layer_refs_for_input(weight)
                case Layer(base=LinearGatherLayerBase(input=input, weight=weight)):
                    yield from self._iter_layer_refs_for_input(input)
                    yield from self._iter_layer_refs_for_input(weight)

    def __call__(self, layers: Mapping[str, OperationSeq | Layer]) -> _ChainGraph:
        v = list(layers.keys())
        succ: dict[str, str] = {}
        ref_gathers: dict[str, GatheredLayers] = {}

        for layer_id, ops in layers.items():
            match ops:
                case OperationSeq(layer_refs=LayerRefs(types=[LayerRefs.TYPE_LAYER], layer_ids=[ref_layer_id])):
                    succ[ref_layer_id] = layer_id
                case OperationSeq(operations=operations):
                    for op in operations:
                        match op:
                            case Linear(
                                weight_ops=OperationSeq(
                                    layer_refs=LayerRefs(types=[LayerRefs.TYPE_LAYER], layer_ids=[ref_layer_id])
                                )
                            ):
                                succ[ref_layer_id] = layer_id
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
                        InputLayerBase(
                            input=GatheredLayers(
                                refs=LayerRefs(types=[LayerRefs.TYPE_LAYER], layer_ids=[ref_layer_id])
                            ) as gl
                        )
                        | LinearLayerBase(
                            input=GatheredLayers(
                                refs=LayerRefs(types=[LayerRefs.TYPE_LAYER], layer_ids=[ref_layer_id])
                            ) as gl
                        )
                        | LinearLayerBase(
                            weight=GatheredLayers(
                                refs=LayerRefs(types=[LayerRefs.TYPE_LAYER], layer_ids=[ref_layer_id])
                            ) as gl
                        )
                        | LinearGatherLayerBase(
                            input=GatheredLayers(
                                refs=LayerRefs(types=[LayerRefs.TYPE_LAYER], layer_ids=[ref_layer_id])
                            ) as gl
                        )
                        | LinearGatherLayerBase(
                            weight=GatheredLayers(
                                refs=LayerRefs(types=[LayerRefs.TYPE_LAYER], layer_ids=[ref_layer_id])
                            ) as gl
                        )
                    )
                ):
                    succ[ref_layer_id] = layer_id
                    ref_gathers[ref_layer_id] = gl

        visited: set[str] = set()

        for ref in self._iter_all_layer_refs(layers):
            if ref not in visited:
                visited.add(ref)
            else:
                succ.pop(ref, None)

        pred: dict[str, str] = {b: a for a, b in succ.items()}

        return _ChainGraph(v, succ, pred, ref_gathers)
