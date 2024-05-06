import copy
from typing import Iterator, Literal, Sequence, get_args

from lib.vectorize.settings import (
    LinearsSymmetriesSettings,
    OptimizeSingleUseGathersPreset,
    OptimizeSingleUseGathersSettings,
    OptimizeTailRefsSettings,
    VectorizeSettings,
    VectorizeSettingsPartial,
)


class _Barrier:
    pass


_BARRIER = _Barrier()


_VARIANTS_PARTIALS: list[list[VectorizeSettingsPartial | _Barrier]] = [
    [
        VectorizeSettingsPartial(transpose_fixed_count_reduce=False),
        VectorizeSettingsPartial(transpose_fixed_count_reduce=True),
    ],
    [
        VectorizeSettingsPartial(optimize_single_use_gathers=OptimizeSingleUseGathersSettings.preset(preset))
        for preset in get_args(OptimizeSingleUseGathersPreset)
    ],
    [
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="never")),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="any")),
    ]
]


_VARIANTS_TEST_OPTIMIZATION_EFFECT_PARTIALS: list[list[VectorizeSettingsPartial | _Barrier]] = [
    [
        VectorizeSettingsPartial(transpose_fixed_count_reduce=False),
        VectorizeSettingsPartial(transpose_fixed_count_reduce=True),
    ],
    [
        VectorizeSettingsPartial(
            iso_compression=False,
            linears_optimize_unique_ref_pairs=False,
            linears_symmetries=False,
            optimize_tail_refs=False,
            optimize_single_use_gathers=False,
        ),
        _BARRIER,
        VectorizeSettingsPartial(iso_compression=False, linears_optimize_unique_ref_pairs=True),
        VectorizeSettingsPartial(iso_compression=True, linears_optimize_unique_ref_pairs=False),
        VectorizeSettingsPartial(iso_compression=True, linears_optimize_unique_ref_pairs=True),
        _BARRIER,
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="never")),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="sided_only")),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="full_only")),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="any")),
        _BARRIER,
        VectorizeSettingsPartial(optimize_tail_refs=OptimizeTailRefsSettings()),
        _BARRIER,
        *[
            VectorizeSettingsPartial(optimize_single_use_gathers=OptimizeSingleUseGathersSettings.preset(preset))
            for preset in get_args(OptimizeSingleUseGathersPreset)
        ],
    ],
]


_VARIANTS_TEST_OPTIMIZATION_EFFECT_PARTIALS_ALL_COMBINATIONS: list[list[VectorizeSettingsPartial | _Barrier]] = [
    [
        VectorizeSettingsPartial(allow_repeat_gathers=False),
        VectorizeSettingsPartial(allow_repeat_gathers=True),
    ],
    [
        VectorizeSettingsPartial(granularize_by_weight=False),
        VectorizeSettingsPartial(granularize_by_weight=True),
    ],
    [
        VectorizeSettingsPartial(transpose_fixed_count_reduce=False),
        VectorizeSettingsPartial(transpose_fixed_count_reduce=True),
    ],
    [
        VectorizeSettingsPartial(iso_compression=False),
        VectorizeSettingsPartial(iso_compression=True),
    ],
    [
        VectorizeSettingsPartial(linears_optimize_unique_ref_pairs=False),
        VectorizeSettingsPartial(linears_optimize_unique_ref_pairs=True),
    ],
    [
        VectorizeSettingsPartial(linears_symmetries=False),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="never")),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="sided_only")),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="full_only")),
        VectorizeSettingsPartial(linears_symmetries=LinearsSymmetriesSettings(pad="any")),
    ],
    [
        VectorizeSettingsPartial(optimize_tail_refs=False),
        VectorizeSettingsPartial(optimize_tail_refs=OptimizeTailRefsSettings()),
    ],
    [
        VectorizeSettingsPartial(optimize_single_use_gathers=False),
        *[
            VectorizeSettingsPartial(optimize_single_use_gathers=OptimizeSingleUseGathersSettings.preset(preset))
            for preset in get_args(OptimizeSingleUseGathersPreset)
        ],
    ],
]


def _iter_all_variants(
    partials: Sequence[Sequence[VectorizeSettingsPartial | _Barrier]],
    base: VectorizeSettingsPartial | None = None,
) -> Iterator[VectorizeSettings]:
    if not partials:
        if base is not None:
            yield VectorizeSettings(**base)
        return

    head, tail = partials[0], partials[1:]

    if base is None:
        base = VectorizeSettingsPartial()

    prev_variant_base: VectorizeSettingsPartial | None = None
    for variant in head:
        if isinstance(variant, _Barrier):
            assert prev_variant_base is not None
            base = prev_variant_base
            continue

        base_this = copy.deepcopy(base)
        base_this.update(copy.deepcopy(variant))
        prev_variant_base = base_this
        yield from _iter_all_variants(tail, base_this)


VectorizeSettingsPresets = Literal[
    "default_only", "tuning", "test_optimizations_effect", "test_optimizations_effect_all_combinations"
]


_VARIANTS_MAP: dict[VectorizeSettingsPresets, list[list[VectorizeSettingsPartial | _Barrier]]] = {
    "default_only": [[VectorizeSettingsPartial()]],
    "tuning": _VARIANTS_PARTIALS,
    "test_optimizations_effect": _VARIANTS_TEST_OPTIMIZATION_EFFECT_PARTIALS,
    "test_optimizations_effect_all_combinations": _VARIANTS_TEST_OPTIMIZATION_EFFECT_PARTIALS_ALL_COMBINATIONS,
}


def iterate_vectorize_settings_presets(choose_presets: VectorizeSettingsPresets) -> Iterator[VectorizeSettings]:
    return _iter_all_variants(_VARIANTS_MAP[choose_presets])


if __name__ == "__main__":
    vs = list(iterate_vectorize_settings_presets("test_optimizations_effect"))
    for v in vs:
        print(v)
        print()

    print(len(vs))
    print(len(set(vs)))
