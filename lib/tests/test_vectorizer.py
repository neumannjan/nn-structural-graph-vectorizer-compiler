from lib.datasets.dataset import BuiltDatasetInstance
from lib.sources.base import LayerDefinition
from lib.sources.minimal_api.internal.java import compute_java_neurons_per_layer

EXPECTED_GSAGE_LAYERS: list[LayerDefinition] = [
    LayerDefinition("node_feature__f", "FactLayer"),
    LayerDefinition("atom_embed__r", "RuleLayer"),
    LayerDefinition("atom_embed__ag", "AggregationLayer"),
    LayerDefinition("atom_embed__wa", "WeightedAtomLayer"),
    LayerDefinition("l1_embed__r", "RuleLayer"),
    LayerDefinition("l1_embed__ag", "AggregationLayer"),
    LayerDefinition("l1_embed__wa", "WeightedAtomLayer"),
    LayerDefinition("l2_embed__r", "RuleLayer"),
    LayerDefinition("l2_embed__ag", "AggregationLayer"),
    LayerDefinition("l2_embed__wa", "WeightedAtomLayer"),
    LayerDefinition("predict__r", "RuleLayer"),
    LayerDefinition("predict__ag", "AggregationLayer"),
    LayerDefinition("predict__wa", "WeightedAtomLayer"),
]

EXPECTED_MUTAG_LAYERS_PRECEDENCE: list[tuple[LayerDefinition, LayerDefinition]] = [
    (LayerDefinition(id="i__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="br__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="f__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="cl__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="n__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="o__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="c__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="h__f", type="FactLayer"), LayerDefinition(id="atom_embed__r", type="RuleLayer")),
    (LayerDefinition(id="b_1__f", type="FactLayer"), LayerDefinition(id="bond_embed__r", type="RuleLayer")),
    (LayerDefinition(id="b_2__f", type="FactLayer"), LayerDefinition(id="bond_embed__r", type="RuleLayer")),
    (LayerDefinition(id="b_3__f", type="FactLayer"), LayerDefinition(id="bond_embed__r", type="RuleLayer")),
    (LayerDefinition(id="b_4__f", type="FactLayer"), LayerDefinition(id="bond_embed__r", type="RuleLayer")),
    (LayerDefinition(id="b_5__f", type="FactLayer"), LayerDefinition(id="bond_embed__r", type="RuleLayer")),
    (LayerDefinition(id="b_7__f", type="FactLayer"), LayerDefinition(id="bond_embed__r", type="RuleLayer")),
    (
        LayerDefinition(id="atom_embed__r", type="RuleLayer"),
        LayerDefinition(id="atom_embed__ag", type="AggregationLayer"),
    ),
    (
        LayerDefinition(id="atom_embed__ag", type="AggregationLayer"),
        LayerDefinition(id="atom_embed__wa", type="WeightedAtomLayer"),
    ),
    (
        LayerDefinition(id="bond_embed__r", type="RuleLayer"),
        LayerDefinition(id="bond_embed__ag", type="AggregationLayer"),
    ),
    (
        LayerDefinition(id="bond_embed__ag", type="AggregationLayer"),
        LayerDefinition(id="bond_embed__wa", type="WeightedAtomLayer"),
    ),
    (
        LayerDefinition(id="atom_embed__wa", type="WeightedAtomLayer"),
        LayerDefinition(id="layer_1__wr", type="WeightedRuleLayer"),
    ),
    (
        LayerDefinition(id="bond_embed__wa", type="WeightedAtomLayer"),
        LayerDefinition(id="layer_1__wr", type="WeightedRuleLayer"),
    ),
    (
        LayerDefinition(id="layer_1__wr", type="WeightedRuleLayer"),
        LayerDefinition(id="layer_1__ag", type="AggregationLayer"),
    ),
    (LayerDefinition(id="layer_1__ag", type="AggregationLayer"), LayerDefinition(id="layer_1__a", type="AtomLayer")),
    (LayerDefinition(id="layer_1__a", type="AtomLayer"), LayerDefinition(id="layer_2__wr", type="WeightedRuleLayer")),
    (
        LayerDefinition(id="bond_embed__wa", type="WeightedAtomLayer"),
        LayerDefinition(id="layer_2__wr", type="WeightedRuleLayer"),
    ),
    (
        LayerDefinition(id="layer_2__wr", type="WeightedRuleLayer"),
        LayerDefinition(id="layer_2__ag", type="AggregationLayer"),
    ),
    (LayerDefinition(id="layer_2__ag", type="AggregationLayer"), LayerDefinition(id="layer_2__a", type="AtomLayer")),
    (
        LayerDefinition(id="bond_embed__wa", type="WeightedAtomLayer"),
        LayerDefinition(id="layer_3__wr", type="WeightedRuleLayer"),
    ),
    (LayerDefinition(id="layer_2__a", type="AtomLayer"), LayerDefinition(id="layer_3__wr", type="WeightedRuleLayer")),
    (
        LayerDefinition(id="layer_3__wr", type="WeightedRuleLayer"),
        LayerDefinition(id="layer_3__ag", type="AggregationLayer"),
    ),
    (LayerDefinition(id="layer_3__ag", type="AggregationLayer"), LayerDefinition(id="layer_3__a", type="AtomLayer")),
    (LayerDefinition(id="layer_3__a", type="AtomLayer"), LayerDefinition(id="predict__r", type="RuleLayer")),
    (LayerDefinition(id="predict__r", type="RuleLayer"), LayerDefinition(id="predict__ag", type="AggregationLayer")),
    (
        LayerDefinition(id="predict__ag", type="AggregationLayer"),
        LayerDefinition(id="predict__wa", type="WeightedAtomLayer"),
    ),
]


def test_layers_topological_ordering1(tu_mutag_gsage: BuiltDatasetInstance):
    _, layers = compute_java_neurons_per_layer(tu_mutag_gsage.samples)

    assert layers == EXPECTED_GSAGE_LAYERS


def test_layers_topological_ordering2(mutag: BuiltDatasetInstance):
    _, layers = compute_java_neurons_per_layer(mutag.samples)

    for a, b in EXPECTED_MUTAG_LAYERS_PRECEDENCE:
        assert layers.index(a) < layers.index(b), f"{a} < {b} in layers"
