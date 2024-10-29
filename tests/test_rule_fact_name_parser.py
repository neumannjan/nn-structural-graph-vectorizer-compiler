import pytest
from compute_graph_vectorize.facts.model import get_rule_or_fact_main_name
from compute_graph_vectorize.facts.parser import parse_rule_or_fact

EXAMPLES: list[tuple[str, str]] = [
    ("<[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> node_feature(16)", "node_feature"),
    ("<[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]> node_feature(15)", "node_feature"),
    ("[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] node_feature(15)", "node_feature"),
    ("l2_embed(23)", "l2_embed"),
    ("atom_embed(10)", "atom_embed"),
    ("{10, 10} l1_embed(X) :- atom_embed(Y), *edge(Y, X). [transformation=identity, aggregation=avg]", "l1_embed"),
    ("{1, 10} predict :- l2_embed(X). [transformation=identity, aggregation=avg]", "predict"),
    ("{10, 10} l1_embed(X) :- atom_embed(X). [transformation=identity]", "l1_embed"),
    ("{10, 10} l2_embed(X) :- l1_embed(Y), *edge(Y, X). [transformation=identity, aggregation=avg]", "l2_embed"),
    ("predict", "predict"),
    ("{10, 7} atom_embed(X) :- node_feature(X). [transformation=identity]", "atom_embed"),
    ("{10, 10} l2_embed(X) :- l1_embed(X). [transformation=identity]", "l2_embed"),
]


@pytest.mark.parametrize("input,expected", EXAMPLES)
def test_main_name(input: str, expected: str):
    actual = get_rule_or_fact_main_name(parse_rule_or_fact(input))

    assert actual == expected
