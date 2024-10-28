from typing import NamedTuple


class Fact(NamedTuple):
    name: str
    terms: list[str]
    shape: tuple[int, ...] | None


class Rule(NamedTuple):
    lhs: Fact
    rhs: tuple[Fact, ...]


def get_rule_or_fact_main_name(fact_or_rule: Rule | Fact):
    match fact_or_rule:
        case Rule(lhs=lhs):
            return lhs.name
        case Fact(name=name):
            return name
