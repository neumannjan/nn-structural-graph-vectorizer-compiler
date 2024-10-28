import itertools
from typing import Iterable, Iterator, Sequence

from compute_graph_vectorize.facts.model import Fact


class TermMap:
    def __init__(self, term_prefix: str, index_provider: Iterator[int]) -> None:
        self._map: dict[str, str] = {}
        self.term_prefix = term_prefix
        self.index_provider = index_provider

    def get_term(self, original_term: str) -> str:
        if original_term in self._map:
            return self._map[original_term]

        self._map[original_term] = out = self.term_prefix + str(next(self.index_provider))
        return out


class DatasetDuplicator:
    def __init__(self, term_prefix: str) -> None:
        self.term_prefix = term_prefix
        self.index_provider = iter(itertools.count())

    def duplicate_samples(self, dataset: Iterable[Iterable[Fact]], times: int = 1, include_orig: bool = False):
        if (times > 1 or include_orig) and not isinstance(dataset, Sequence):
            dataset = [[f for f in s] for s in dataset]

        if include_orig:
            yield from dataset
            times -= 1

        for _ in range(times):
            yield from self._duplicate_samples(dataset)

    def _duplicate_samples(self, dataset: Iterable[Iterable[Fact]]):
        term_map = TermMap(self.term_prefix, self.index_provider)
        for sample in dataset:
            yield self.duplicate_sample(sample, term_map)

    def duplicate_sample(self, sample: Iterable[Fact], term_map: TermMap):
        for fact in sample:
            yield self.duplicate_fact(fact, term_map)

    def duplicate_fact(self, fact: Fact, term_map: TermMap) -> Fact:
        return Fact(name=fact.name, terms=[term_map.get_term(t) for t in fact.terms])
