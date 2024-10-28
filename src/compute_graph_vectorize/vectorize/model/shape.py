from typing import Iterable, Sequence, overload

from compute_graph_vectorize.vectorize.model.repr import repr_slots


class VariousShape:
    __slots__ = ()
    __repr__ = repr_slots

    def __hash__(self) -> int:
        return hash(())

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, VariousShape)


class AnyShape:
    __slots__ = ()
    __repr__ = repr_slots

    def __hash__(self) -> int:
        return hash(())

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, VariousShape)


class ConcreteShape(Sequence[int]):
    __slots__ = ("dims",)
    __match_args__ = ("dims",)

    def __init__(self, dims: Iterable[int]) -> None:
        self.dims = tuple(dims)

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> "ConcreteShape": ...

    def __getitem__(self, key: int | slice) -> "int | ConcreteShape":
        if isinstance(key, int):
            return self.dims[key]
        elif isinstance(key, slice):
            return ConcreteShape(self.dims[key])
        else:
            assert False

    def __len__(self) -> int:
        return len(self.dims)

    def __hash__(self) -> int:
        return hash(self.dims)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, ConcreteShape) and self.dims == value.dims

    def __repr__(self) -> str:
        return f"[{', '.join((str(d) for d in self.dims))}]"


Shape = VariousShape | AnyShape | ConcreteShape
