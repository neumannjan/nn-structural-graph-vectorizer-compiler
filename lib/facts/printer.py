from typing import Iterable, TextIO

from lib.facts.model import Fact


class DatasetPrinter:
    def __init__(self, out: TextIO) -> None:
        self._out = out

    def print_fact(self, fact: Fact):
        self._out.write(fact.name + '(' + ', '.join(fact.terms) + ')')

    def print_sample(self, sample: Iterable[Fact]):
        it = iter(sample)

        try:
            fact = next(it)
            self.print_fact(fact)
        except StopIteration:
            return

        try:
            while True:
                fact = next(it)
                self._out.write(', ')
                self.print_fact(fact)
        except StopIteration:
            pass

        self._out.write('.\n')

    def print_dataset(self, samples: Iterable[Iterable[Fact]]):
        for sample in samples:
            self.print_sample(sample)
