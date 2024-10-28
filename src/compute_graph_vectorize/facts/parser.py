import re
from collections import deque
from typing import Generic, Iterable, Iterator, Literal, TextIO, TypeVar

from compute_graph_vectorize.facts.model import Fact, Rule

_TOKENIZER_REGEX = re.compile(r"(\d+\.\d+|[,\(\)\.{}]|\w+|:-|<?\[|\]>?)", re.IGNORECASE | re.UNICODE)
_RESERVED_TOKENS = {",", "(", ")", ".", "{", "}", ":-", "<[", "]>", "[", "]"}


def tokenize(sample: str):
    return _TOKENIZER_REGEX.finditer(sample)


def parse_file(fp: TextIO):
    fp.seek(0)
    return parse_samples(fp)


def parse_samples(samples: Iterable[str]):
    for sample in samples:
        yield parse_sample(sample)


_T = TypeVar("_T")


class _MultiIter(Generic[_T]):
    __slots__ = ("iterators",)

    def __init__(self, it: Iterator[_T]):
        self.iterators = deque([it])

    def next(self) -> _T:
        while True:
            try:
                it = self.iterators[0]
                return next(it)
            except IndexError:
                raise StopIteration()
            except StopIteration:
                self.iterators.popleft()

    def prepend(self, it: Iterator[_T]):
        self.iterators.appendleft(it)

    def prepend_value(self, value: _T):
        self.iterators.appendleft(iter((value,)))


class ParserError(Exception):
    pass


def parse_sample(sample: str):
    tokens_iter: _MultiIter[str] = _MultiIter((t.group(0) for t in tokenize(sample)))

    state = 0
    # 0: before fact - expecting fact (starting with fact name or opening shape brace) or end
    # 1: after fact - expecting fact separator or end

    try:
        while True:
            token = tokens_iter.next()

            if state == 0:
                if token == ".":
                    break
                if token == "{":
                    tokens_iter.prepend_value(token)
                    yield _parse_fact(tokens_iter)
                    state = 1
                elif token in _RESERVED_TOKENS:
                    raise ParserError()
                else:
                    tokens_iter.prepend_value(token)
                    yield _parse_fact(tokens_iter)
                    state = 1
            elif state == 1:
                if token == ",":
                    state = 0
                elif token == ".":
                    break
                else:
                    raise ParserError()
    except StopIteration:
        pass


def _parse_fact(tokens_iter: _MultiIter[str], is_rule_lhs=False, is_rule_rhs=False) -> Fact:
    # 0: beginning - expecting name or shape opening brace or value opening brace
    # 1: after shape - expecting name
    # 2: after name - expecting opening parenthesis or end
    # 3: inside shape, before number - expecting number
    # 4: inside shape, after number - expecting separator or shape close
    # 5: before term - expecting term or end (fact closing parenthesis)
    # 6: after term - expecting term separator or closing parenthesis
    # 7: in value - ignoring anything until value closing brace, then going to state 'after shape'

    state: Literal[0, 1, 2, 3, 4, 5, 6, 7] = 0

    terms = []

    name: str = ""
    shape: list[int] = []

    try:
        while True:
            token = tokens_iter.next()

            if state == 0:
                if token == "{":
                    state = 3
                elif token == "<[" or token == "[":
                    state = 7
                elif token in _RESERVED_TOKENS:
                    raise ParserError()
                else:
                    name = token
                    state = 2
            elif state == 1:
                if token in _RESERVED_TOKENS:
                    raise ParserError()
                else:
                    name = token
                    state = 2
            elif state == 2:
                if token == "(":
                    state = 5
                elif is_rule_lhs and token == ":-":
                    tokens_iter.prepend_value(token)
                    return Fact(name, terms, tuple(shape))
                elif is_rule_rhs and token in (",", "."):
                    tokens_iter.prepend_value(token)
                    return Fact(name, terms, tuple(shape))
                else:
                    raise ParserError()
            elif state == 3:
                if token in _RESERVED_TOKENS:
                    raise ParserError()
                else:
                    num = int(token)
                    shape.append(num)
                    state = 4
            elif state == 4:
                if token == ",":
                    state = 3
                elif token == "}":
                    state = 1
                else:
                    raise ParserError()
            elif state == 5:
                if token == ")":
                    return Fact(name, terms, tuple(shape))
                elif token in _RESERVED_TOKENS:
                    raise ParserError()
                else:
                    terms.append(token)
                    state = 6
            elif state == 6:
                if token == ",":
                    state = 5
                elif token == ")":
                    return Fact(name, terms, tuple(shape))
                else:
                    raise ParserError()
            elif state == 7:
                if token == "]>" or token == "]":
                    state = 1
    except StopIteration:
        if state == 2:
            return Fact(name, terms, tuple(shape))
        else:
            raise ParserError()


def parse_fact(fact: str) -> Fact:
    tokens_iter = _MultiIter((t.group(0) for t in tokenize(fact)))
    return _parse_fact(tokens_iter)


def parse_rule_or_fact(rule: str) -> Rule | Fact:
    tokens_iter = _MultiIter((t.group(0) for t in tokenize(rule)))

    state: Literal[0, 1, 2, 3] = 0
    # 0: start - expecting single fact as lhs
    # 1: after lhs - expecting :- or end
    # 2: rhs - expecting fact
    # 3: rhs after fact - expecting comma or dot or end

    lhs: Fact | None = None
    rhs: list[Fact] = []

    try:
        while True:
            if state == 0:
                lhs = _parse_fact(tokens_iter, is_rule_lhs=True)
                state = 1
            elif state == 1:
                token = tokens_iter.next()
                if token == ":-":
                    state = 2
                elif token == ".":
                    raise StopIteration()
                else:
                    raise ValueError()
            elif state == 2:
                _fact = _parse_fact(tokens_iter)
                rhs.append(_fact)
                state = 3
            elif state == 3:
                token = tokens_iter.next()
                if token == ",":
                    state = 2
                elif token == ".":
                    raise StopIteration()
                else:
                    raise ValueError()
    except StopIteration:
        if state == 1:
            assert lhs is not None
            return lhs
        elif state == 3:
            assert lhs is not None
            assert len(rhs) > 0
            return Rule(lhs, tuple(rhs))
        else:
            raise ValueError()
