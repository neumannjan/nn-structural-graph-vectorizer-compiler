import warnings
from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from lib.utils import (
    detect_repeating_interleaved_sequence_in_list,
    detect_repeating_K_sequence_in_list,
    detect_repeating_sequence_in_list,
)
from lib.vectorize.model import *
from lib.vectorize.model.layer import DimensionLift, DimensionLifts, lifts_dimension_match
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation

_TSeq = TypeVar("_TSeq", list, Refs)


_TInput = TypeVar("_TInput", bound=Input)
_SInput = TypeVar("_SInput", bound=Input)


class LiftSymmetricalLinears(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._counts = ComputeLayerCounts(network)

    def _get_refs_as_numpy(self, input: Refs) -> np.ndarray:
        return np.stack([input.types, input.layer_ids, input.ordinals], axis=1)

    def detect_dim_lift_single(self, refs: GatheredLayers | Refs, preferred_period: int | None) -> DimensionLift | None:
        match refs:
            case Refs():
                arr = self._get_refs_as_numpy(refs)
            case GatheredLayers(gather=GenericGather(ordinals)):
                arr = np.array(ordinals)
            case GatheredLayers(gather=NoopGather()):
                return None
            case _:
                raise ValueError(refs.gather)

        if preferred_period is not None:
            out_period = detect_repeating_K_sequence_in_list(arr, period=preferred_period, allow_last_incomplete=False)
            if out_period is not None:
                return (-1, out_period)

        out_period = detect_repeating_sequence_in_list(arr, allow_last_incomplete=False)
        if out_period is not None:
            return (-1, out_period)

        out_repeats = detect_repeating_interleaved_sequence_in_list(arr, allow_last_incomplete=False)
        if out_repeats is not None:
            out_period = len(arr) // out_repeats
            return (out_period, -1)

        return None

    def detect_dim_lift(
        self,
        batch_id: int,
        input: _TInput,
        weight: _SInput,
        existing_lifts: DimensionLifts,
        preferred_period: int | None,
    ) -> tuple[DimensionLifts, tuple[_TInput, _SInput]]:
        input_count = self._counts.compute_input_count(batch_id, input)
        weight_count = self._counts.compute_input_count(batch_id, weight)

        if existing_lifts is not None:
            i_dim_lift, w_dim_lift = existing_lifts

            if lifts_dimension_match(existing_lifts):
                if input_count < weight_count:
                    w_dim_lift = None
                elif weight_count < input_count:
                    i_dim_lift = None
                else:
                    warnings.warn("Possibly unoptimal lifts. Will likely use unnecessary View modules.")
                    return existing_lifts, (input, weight)

        else:
            i_dim_lift, w_dim_lift = None, None

        i_dim_lift = i_dim_lift or self.detect_dim_lift_single(input, preferred_period=preferred_period)
        w_dim_lift = w_dim_lift or self.detect_dim_lift_single(weight, preferred_period=preferred_period)

        expected_count = self._counts.compute_linear_count(batch_id, input, weight, existing_lifts)

        input_lifted = self._get_dim_lifted_input(input, i_dim_lift) if i_dim_lift is not None else input
        weight_lifted = self._get_dim_lifted_input(weight, w_dim_lift) if w_dim_lift is not None else weight

        if i_dim_lift is not None and w_dim_lift is not None and not lifts_dimension_match((i_dim_lift, w_dim_lift)):
            # try both
            both_lifted_count = self._counts.compute_linear_count(
                batch_id, input_lifted, weight_lifted, (i_dim_lift, w_dim_lift)
            )
            if both_lifted_count == expected_count:
                return (i_dim_lift, w_dim_lift), (input_lifted, weight_lifted)

        input_lifted_count = (
            self._counts.compute_linear_count(batch_id, input_lifted, weight, (i_dim_lift, i_dim_lift))
            if i_dim_lift is not None
            else None
        )

        weight_lifted_count = (
            self._counts.compute_linear_count(batch_id, input, weight_lifted, (w_dim_lift, w_dim_lift))
            if w_dim_lift is not None
            else None
        )

        if i_dim_lift is not None and w_dim_lift is not None:
            # can only do one or the other
            if input_lifted_count == expected_count and weight_lifted_count == expected_count:
                # must choose the cheaper one of the two
                input_refs = self._input_to_refs(input)
                weight_refs = self._input_to_refs(weight)
                input_lifted_refs = self._input_to_refs(input_lifted)
                weight_lifted_refs = self._input_to_refs(weight_lifted)

                if len(input_lifted_refs) + len(weight_refs) <= len(input_refs) + len(weight_lifted_refs):
                    # input
                    w_dim_lift = None
                else:
                    # weight
                    i_dim_lift = None
            elif input_lifted_count == expected_count:
                # input
                w_dim_lift = None
            elif weight_lifted_count == expected_count:
                # weight
                i_dim_lift = None
            else:
                # nothing
                i_dim_lift, w_dim_lift = None, None

        if i_dim_lift is not None and input_lifted_count == expected_count:
            return (i_dim_lift, i_dim_lift), (input_lifted, weight)
        elif w_dim_lift is not None and weight_lifted_count == expected_count:
            return (w_dim_lift, w_dim_lift), (input, weight_lifted)

        return None, (input, weight)

    def _get_dim_lifted_refs(self, refs: _TSeq, dim_lift: DimensionLift) -> _TSeq:
        match dim_lift:
            case (period, -1):
                refs = refs[:: len(refs) // period]
            case (-1, period):
                refs = refs[:period]
            case _:
                assert False, f"{dim_lift}"

        return refs

    def _get_dim_lifted_input(self, input: Input, dim_lift: DimensionLift) -> Input:
        match input:
            case Refs():
                return self._get_dim_lifted_refs(input, dim_lift)
            case GatheredLayers(refs=refs, gather=GenericGather(ordinals)):
                return GatheredLayers(refs=refs, gather=GenericGather(self._get_dim_lifted_refs(ordinals, dim_lift)))
            case _:
                raise ValueError(input)

    def _input_to_refs(self, input: Input) -> Sequence:
        match input:
            case Refs():
                return input
            case GatheredLayers(gather=GenericGather(ordinals)):
                return ordinals
            case _:
                raise ValueError(input)

    def _simplify(
        self,
        batch_id: int,
        base: LinearLayerBase | LinearGatherLayerBase,
        input: Input,
        weight: Input,
        preferred_period: int | None,
        existing_lifts: DimensionLifts,
    ):
        lifts, (input, weight) = self.detect_dim_lift(batch_id, input, weight, existing_lifts, preferred_period)
        base.input = input
        base.weight = weight
        base.lifts = lifts

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        match layer:
            case Layer(base=InputLayerBase()):
                pass
            case Layer(
                base=(
                    LinearLayerBase(input=input, weight=weight, lifts=lifts)
                    | LinearGatherLayerBase(input=input, weight=weight, lifts=lifts)
                ) as base,
                aggregate=FixedCountReduce(period=period),
            ):
                self._simplify(batch, base, input, weight, preferred_period=period, existing_lifts=lifts)
            case Layer(
                base=(
                    LinearLayerBase(input=input, weight=weight, lifts=lifts)
                    | LinearGatherLayerBase(input=input, weight=weight, lifts=lifts)
                ) as base
            ):
                self._simplify(batch, base, input, weight, preferred_period=None, existing_lifts=lifts)
            case _:
                assert False, f"{layer}"
        return layer

    def lift_symmetrical_linears(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                self(bid, lid, layer)


def lift_symmetrical_linears(network: VectorizedLayerNetwork):
    LiftSymmetricalLinears(network).lift_symmetrical_linears()
    return network
