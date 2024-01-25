import re
from typing import Iterable, Iterator, Match, TextIO

from lib.facts.model import Fact


class DatasetParser:
    TOKENIZER_REGEX = re.compile(r"([,\(\)\.]|\w+)", re.IGNORECASE)

    def parse_file(self, fp: TextIO):
        fp.seek(0)
        return self.parse_samples(fp)

    def parse_samples(self, samples: Iterable[str]):
        for sample in samples:
            yield self.parse_sample(sample)

    def parse_sample(self, sample: str):
        tokens_iter = self.tokenize_sample(sample)

        state = 0
        # 0 expecting fact or end
        # 1 expecting fact separator or end

        try:
            while True:
                token = next(tokens_iter).group(0)

                if state == 0:
                    if token in ('(', ')', ','):
                        raise ValueError()
                    elif token == '.':
                        break
                    else:
                        yield self.parse_fact(token, tokens_iter)
                        state = 1
                elif state == 1:
                    if token == ',':
                        state = 0
                    elif token == '.':
                        break
                    else:
                        raise ValueError()
        except StopIteration:
            pass

    def parse_fact(self, first_token: str, tokens_iter: Iterator[Match[str]]) -> Fact:
        name = first_token

        state = 0
        # 0 expecting opening brace
        # 1 expecting term or end
        # 2 expecting term separator or end

        terms = []

        try:
            while True:
                token = next(tokens_iter).group(0)

                if state == 0:
                    assert token == '('
                    state = 1
                elif state == 1:
                    if token == ')':
                        return Fact(name, terms)
                    elif token in ('.', ',', '('):
                        raise ValueError()
                    else:
                        terms.append(token)
                        state = 2
                elif state == 2:
                    if token == ',':
                        state = 1
                    elif token == ')':
                        return Fact(name, terms)
                    else:
                        raise ValueError()
        except StopIteration:
            raise ValueError()

    def tokenize_sample(self, sample: str):
        return self.TOKENIZER_REGEX.finditer(sample)
