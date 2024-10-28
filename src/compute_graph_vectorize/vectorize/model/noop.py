from compute_graph_vectorize.vectorize.model.repr import repr_slots


class Noop:
    __slots__ = ()
    __repr__ = repr_slots

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, Noop)

    def __hash__(self) -> int:
        return hash(())
