"""Regression tests for POMDP table headers, identifiers, and PPS conversion."""

import pytest

from pomdpy.parsers import pomdp as pomdp_parser


@pytest.mark.parametrize("separator", ["", " ", "\n", " # header comment\n "])
def test_table_letters_can_also_name_states_actions_and_observations(separator):
    model = pomdp_parser.parse(f"""
states: O T
actions: O T
observations: O T O_start T_end
prio 0: O
prio 1: T
start include: O
T{separator}:O
1 0
0 1
T{separator}:T
0 1
1 0
O{separator}:O
1 0 0 0
0 1 0 0
O{separator}:T
0 0 1 0
0 0 0 1
""")

    assert model.states == ["O", "T"]
    assert model.actions == ["O", "T"]
    assert model.obs == ["O", "T", "O_start", "T_end"]
    assert model.start == {0: 1.0}
    assert model.prio == {0: {0}, 1: {1}}
    positive_transitions = {
        state: {
            action: {outcome: prob for outcome, prob in row.items() if prob > 0}
            for action, row in actions.items()
        }
        for state, actions in model.transitions.items()
    }
    assert positive_transitions == {
        0: {0: {(0, 0): 1.0}, 1: {(1, 3): 1.0}},
        1: {0: {(1, 1): 1.0}, 1: {(0, 2): 1.0}},
    }
    assert pomdp_parser.parse(model.to_pomdp_string()).__dict__ == model.__dict__


@pytest.mark.parametrize("observations", ["seen, hidden", " seen, hidden, seen , hidden "])
def test_pps_conversion_preserves_observation_assignments(tmp_path, observations):
    from scripts.convert_pps_to_pomdp import convert_pps_file

    source = tmp_path / "model.txt"
    content = f"""NAME: [example] #
OBSERVATIONS: [{observations}] #
STATES: [{{0;0;seen;T}}, {{1;1;hidden;F}}] #
ACTIONS: [step] #
TRANSITIONS: [0,step,{{0;1}}] #
TRANSITIONS: [1,step,{{0}}] #
"""
    source.write_text(content)
    output = tmp_path / "model.pomdp"
    assert convert_pps_file(source, output) == output
    assert source.read_text() == content

    model = pomdp_parser.parse(output.read_text())
    assert model.obs == ["seen", "hidden"]
    assert model.states == ["s0", "s1"]
    assert model.actions == ["step"]
    assert model.start == {0: 1.0}
    assert model.prio == {0: {0}, 1: {1}}
    positive_transitions = {
        state: {outcome: prob for outcome, prob in actions[0].items() if prob > 0}
        for state, actions in model.transitions.items()
    }
    assert positive_transitions == {
        0: {(0, 0): 0.5, (1, 1): 0.5},
        1: {(0, 0): 1.0},
    }
