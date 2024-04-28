from lib.vectorize.settings import VectorizeSettings

D = VectorizeSettings()


# assert that it matches what I think it does
# allows me to immediately see where I'm using the default values, as well as what the values are
def _D(a, b):
    assert a == b
    return a


SETTINGS_PARAMS = [
    VectorizeSettings(
        linears_optimize_repeating_seq=_D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=_D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=_D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=not _D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=_D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=_D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=_D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=not _D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=_D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=_D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=_D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=not _D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=_D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=not _D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=_D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=not _D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=_D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=_D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=_D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=not _D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=_D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=not _D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=not _D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=not _D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=not _D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=_D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=not _D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=_D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=not _D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
    VectorizeSettings(
        linears_optimize_repeating_seq=not _D(D.linears_optimize_repeating_seq, True),
        optimize_tail_refs=not _D(D.optimize_tail_refs, True),
        linears_optimize_unique_ref_pairs=not _D(D.linears_optimize_unique_ref_pairs, True),
        linears_optimize_unique_ref_pairs_aggressively=_D(D.linears_optimize_unique_ref_pairs_aggressively, False),
    ),
]
