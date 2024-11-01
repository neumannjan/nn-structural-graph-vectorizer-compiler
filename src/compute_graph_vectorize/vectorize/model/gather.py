from compute_graph_vectorize.vectorize.model.repr import repr_slots


class GenericGather:
    __slots__ = ("ordinals",)
    __match_args__ = ("ordinals",)
    __repr__ = repr_slots

    def __init__(self, ordinals: list[int]) -> None:
        self.ordinals = ordinals

    def __hash__(self) -> int:
        return hash(tuple(self.ordinals))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, GenericGather)
            and len(self.ordinals) == len(value.ordinals)
            and all((a == b for a, b in zip(self.ordinals, value.ordinals)))
        )


class TakeSingleValue:
    __slots__ = ("ordinal",)
    __match_args__ = ("ordinal",)
    __repr__ = repr_slots

    def __init__(self, ordinal: int) -> None:
        self.ordinal = ordinal

    def __hash__(self) -> int:
        return hash(self.ordinal)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, TakeSingleValue) and self.ordinal == value.ordinal


class NoopGather:
    __slots__ = ()
    __repr__ = repr_slots


class SliceValues:
    __slots__ = ("start", "end", "step")
    __repr__ = repr_slots

    def __init__(self, start: int, end: int, step: int) -> None:
        self.start = start
        self.end = end
        self.step = step

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.step))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, SliceValues)
            and self.start == value.start
            and self.end == value.end
            and self.step == value.step
        )


class Repeat:
    __slots__ = ("times", "total_length")
    __repr__ = repr_slots

    def __init__(self, times: int, total_length: int) -> None:
        self.times = times
        self.total_length = total_length

    def __hash__(self) -> int:
        return hash((self.times, self.total_length))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, Repeat) and self.times == value.times and self.total_length == value.total_length


class RepeatInterleave:
    __slots__ = ("times", "total_length")
    __repr__ = repr_slots

    def __init__(self, times: int, total_length: int) -> None:
        self.times = times
        self.total_length = total_length

    def __hash__(self) -> int:
        return hash((self.times, self.total_length))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, RepeatInterleave)
            and self.times == value.times
            and self.total_length == value.total_length
        )


OneGather = GenericGather | TakeSingleValue | NoopGather | SliceValues | Repeat | RepeatInterleave


class GatherPair:
    __slots__ = ("a", "b")
    __match_args__ = ("a", "b")
    __repr__ = repr_slots

    def __init__(self, a: "Gather", b: "OneGather") -> None:
        self.a = a
        self.b = b

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, GatherPair) and self.a == value.a and self.b == value.b


Gather = GenericGather | TakeSingleValue | NoopGather | SliceValues | Repeat | RepeatInterleave | GatherPair
