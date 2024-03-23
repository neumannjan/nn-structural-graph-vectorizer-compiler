from lib.vectorize.model import *


def _compute_fact_count(fact: Fact) -> int:
    match fact:
        case UnitFact():
            return 1
        case ValueFact(value=value):
            return value.shape[0]
        case _:
            assert False


class ComputeLayerCounts:
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network

    def compute_weight_count(self, weight: LearnableWeight) -> int:
        return weight.value.shape[0]

    def compute_gather_count(self, in_count: int, gather: Gather) -> int:
        match gather:
            case GenericGather(ordinals=ordinals):
                return len(ordinals)
            case TakeSingleValue(ordinal=_):
                return 1
            case NoopGather():
                return in_count
            case SliceValues(start=start, end=end, step=step):
                return -(-(end - start) // step)
            case Repeat(times=_, total_length=total_length):
                return total_length
            case ViewWithPeriod(period=period):
                return in_count // period
            case GatherPair(a, b):
                count = in_count
                count = self.compute_gather_count(count, a)
                count = self.compute_gather_count(count, b)
                return count
            case _:
                assert False

    def compute_aggregation_count(self, in_count: int, agg: Reduce) -> int:
        match agg:
            case FixedCountReduce(period=period, reduce=_):
                return in_count // period
            case DimReduce(dim=dim, reduce=_):
                assert dim > 0
                return in_count
            case UnevenReduce(counts=counts, reduce=_):
                return len(counts)
            case Noop():
                return in_count
            case _:
                assert False

    def compute_ref_count(self, batch: int, ref: Ref) -> int:
        match ref:
            case FactRef(id=id, ordinal=ordinal):
                return _compute_fact_count(self.network.fact_layers[id].facts[ordinal])
            case NeuronRef(id=id, ordinal=ordinal):
                return 1
            case WeightRef(id=id):
                return self.compute_weight_count(self.network.weights[id])
            case _:
                assert False, f"{ref}"

    def compute_refs_count(self, batch: int, refs: Refs) -> int:
        return sum((self.compute_ref_count(batch, ref) for ref in refs.refs))

    def compute_layer_ref_count(self, batch: int, ref: LayerRef) -> int:
        match ref:
            case FactLayerRef(id=id):
                cnt = self.network.fact_layers[id].count
                assert cnt is not None
                return cnt
            case NeuronLayerRef(id=id):
                cnt = self.network.batches[batch].layers[id].count
                assert cnt is not None
                return cnt
            case WeightRef(id=id):
                return self.compute_weight_count(self.network.weights[id])
            case _:
                assert False, f"{ref}"

    def compute_layer_refs_count(self, batch: int, refs: LayerRefs) -> int:
        return sum((self.compute_layer_ref_count(batch, ref) for ref in refs.refs))

    def compute_input_count(self, batch: int, input: Input) -> int:
        match input:
            case Refs(_) as refs:
                count = self.compute_refs_count(batch, refs)
            case GatheredLayers(refs=refs, gather=gather):
                count = self.compute_layer_refs_count(batch, refs)
                count = self.compute_gather_count(count, gather)
            case _:
                assert False, f"{input}"
        return count

    def compute_linear_count(self, batch: int, input: Input, weight: Input) -> int:
        weight_count = self.compute_input_count(batch, weight)
        input_count = self.compute_input_count(batch, input)

        return max(weight_count, input_count)

    def compute_layer_base_count(self, batch: int, base: LayerBase) -> int:
        match base:
            case InputLayerBase(input=gathered_source):
                return self.compute_input_count(batch, gathered_source)
            case LinearLayerBase(input=input, weight=weight):
                return self.compute_linear_count(batch, input, weight)
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather):
                count = self.compute_linear_count(batch, input, weight)
                count = self.compute_gather_count(count, gather)
                return count
            case _:
                assert False

    def compute_layer_count(self, batch: int, layer: Layer) -> int:
        count = self.compute_layer_base_count(batch, layer.base)
        count = self.compute_aggregation_count(count, layer.aggregate)
        return count

    def compute_counts(self):
        for layer in self.network.fact_layers.values():
            layer.count = sum((_compute_fact_count(f) for f in layer.facts))

        for bid, batch in self.network.batches.items():
            try:
                for lid, layer in batch.layers.items():
                    layer.count = self.compute_layer_count(bid, layer)
            except Exception as e:
                raise Exception(f"Exception in batch {bid}, layer {lid}") from e


def compute_layer_counts(network: VectorizedNetwork):
    ComputeLayerCounts(network).compute_counts()
    return network