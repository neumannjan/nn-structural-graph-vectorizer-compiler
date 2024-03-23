from lib.vectorize.model import *


class ReshapeFixedCountReduce:
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network

    def _get(self, reduce: Reduce) -> tuple[int | None, Reduce]:
        match reduce:
            case FixedCountReduce(period=period, reduce=r):
                return period, DimReduce(dim=1, reduce=r)
            case _:
                return None, reduce

    def _apply_period_to_gather(self, gather: Gather, period: int) -> Gather:
        return GatherPair(gather, ViewWithPeriod(period))

    def _apply_period_to_input(self, input: GatheredLayers, period: int):
        input.gather = self._apply_period_to_gather(input.gather, period)

    def _apply_period(self, base: LayerBase, period: int) -> LayerBase:
        match base:
            case InputLayerBase(input=GatheredLayers() as input):
                self._apply_period_to_input(input, period)
                return base
            case LinearLayerBase(input=input, weight=weight):
                return LinearGatherLayerBase(input=input, weight=weight, gather=ViewWithPeriod(period))
            case LinearGatherLayerBase():
                base.gather = self._apply_period_to_gather(base.gather, period)
                return base
            case _:
                assert False

    def reshape_fixed_count_reduce(self):
        for batch in self.network.batches.values():
            for layer in batch.layers.values():
                period, layer.aggregate = self._get(layer.aggregate)

                if period is not None:
                    layer.base = self._apply_period(layer.base, period)


def reshape_fixed_count_reduce(network: VectorizedNetwork):
    ReshapeFixedCountReduce(network).reshape_fixed_count_reduce()
    return network
