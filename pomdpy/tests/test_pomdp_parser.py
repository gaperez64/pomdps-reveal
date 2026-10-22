"""Regression tests for POMDP table headers, identifiers, and PPS conversion."""

import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest
from lark.exceptions import UnexpectedInput

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.pomdp import POMDP, AtomicPropPOMDP, ParityPOMDP


@pytest.mark.parametrize("specification, model_type", [
    ("", POMDP),
    ("prio 0: left\nprio 1: right", ParityPOMDP),
    ("atom 0: hidden", AtomicPropPOMDP),
])
def test_numeric_matrix_formats_and_comments(specification, model_type):
    model = pomdp_parser.parse(f"""
{specification}
states: left right
actions: move
observations: seen hidden
start: 0 1e0
T: move
2.5e-1 # A matrix row may continue after a comment.
.75
5E-1 0.50
O: move
1. 0
1e-300 1
""")
    assert type(model) is model_type
    assert model.start == {1: 1.0}
    assert model.T == {0: {0: {0: 0.25, 1: 0.75}}, 1: {0: {0: 0.5, 1: 0.5}}}
    assert model.O == {0: {0: {0: 1.0}, 1: {0: 1e-300, 1: 1.0}}}
    assert model.transitions[1][0] == {(0, 0): 0.5, (1, 0): 5e-301, (1, 1): 0.5}
    assert pomdp_parser.parse(model.to_pomdp_string()).__dict__ == model.__dict__


@pytest.mark.parametrize("declaration, attribute", [("prio", "prio"), ("atom", "atoms")])
def test_multidigit_declarations_preserve_distinct_numbers(declaration, attribute):
    model = pomdp_parser.parse(f"""
states: left middle right
actions: move
observations: left middle right
{declaration} 1: left
{declaration} 10: middle
{declaration} 123: right
start: 1 0 0
T: move identity
O: move
1 0 0
0 1 0
0 0 1
""")
    assert getattr(model, attribute) == {1: {0}, 10: {1}, 123: {2}}
    if declaration == "prio":
        assert model.prioinv == {0: 1, 1: 10, 2: 123}
    assert pomdp_parser.parse(model.to_pomdp_string()).__dict__ == model.__dict__


@pytest.mark.parametrize("matrix", [
    ".25 .75 .5 .5",
    ".25\t.75\r\n.5\t.5",
    ".25 # Ignore numbers in comments: 0.99\n.75\n.5# another comment\n.5",
    ".25.75\n.5.5",
    ".25\f.75\r.5\n.5",
])
def test_matrix_layout_preserves_values_and_numeric_identifiers(matrix):
    model = pomdp_parser.parse(f"""
states: 2
actions: 1
observations: 2
start include: 0
T: 0 {matrix}
O: 0 {matrix}
""")
    assert model.states == ["0", "1"]
    assert model.actions == ["0"]
    assert model.obs == ["0", "1"]
    assert model.start == {0: 1.0}
    assert model.T == {0: {0: {0: 0.25, 1: 0.75}}, 1: {0: {0: 0.5, 1: 0.5}}}
    assert model.O == {0: {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}}


@pytest.mark.parametrize("header, rule", [("T", "trans_entry"), ("O", "obs_spec_tail")])
def test_single_probabilities_after_numeric_identifiers(header, rule):
    tree = pomdp_parser.parser.parse(f"""
states: 2
actions: 1
observations: 2
start: uniform
{header}: 0 : 0 : 1 .5
""")
    assert next(tree.find_data(rule)).children == ["0", "0", "1", 0.5]


@pytest.mark.parametrize("probability, error", [
    ("1.1", AssertionError),
    ("1e309", AssertionError),
    ("-0.1", UnexpectedInput),
    ("1e-", UnexpectedInput),
    ("0.5x", UnexpectedInput),
    ("nan", UnexpectedInput),
])
def test_invalid_probability_does_not_affect_the_next_parse(probability, error):
    source = """
states: left right
actions: move
observations: seen hidden
start: 0 1
T: move 1 0 0 {probability}
O: move 1 0 0 1
"""
    with pytest.raises(error):
        pomdp_parser.parse(source.format(probability=probability))
    model = pomdp_parser.parse(source.format(probability="1"))
    assert type(model) is POMDP
    assert model.start == {1: 1.0}
    assert model.transitions == {0: {0: {(0, 0): 1.0}}, 1: {0: {(1, 1): 1.0}}}


@pytest.mark.parametrize("matrix", ["1 0 0", "1 0 0 1 0"])
def test_incorrect_matrix_lengths_are_rejected(matrix):
    with pytest.raises(AssertionError):
        pomdp_parser.parse(f"""
states: left right
actions: move
observations: seen
start: uniform
T: move {matrix}
O: move uniform
""")


@pytest.mark.parametrize("declaration", ["states", "actions", "observations"])
def test_numeric_declarations_create_indexed_identifiers(declaration):
    names = {"states": ["left", "right"], "actions": ["wait", "move"],
             "observations": ["seen", "hidden"]}
    headers = "\n".join(
        f"{key}: {2 if key == declaration else ' '.join(values)}"
        for key, values in names.items()
    )
    model = pomdp_parser.parse(headers + """
start: uniform
T: * uniform
O: * uniform
""")
    names[declaration] = ["0", "1"]
    actual_names = {"states": model.states, "actions": model.actions,
                    "observations": model.obs}
    assert actual_names == names
    assert model.start == {0: 0.5, 1: 0.5}
    assert model.transitions == {
        state: {action: {(0, 0): 0.25, (0, 1): 0.25,
                         (1, 0): 0.25, (1, 1): 0.25}
                for action in range(2)}
        for state in range(2)
    }


@pytest.mark.parametrize("start, expected", [
    ("start: uniform", {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}),
    ("start: reset", {0: 1.0}),
    ("start exclude: middle", {0: 0.5, 2: 0.5}),
    ("start include: 0", {0: 1.0}),
    ("start include: 0 2", {0: 0.5, 2: 0.5}),
    ("start include: left 2", {0: 0.5, 2: 0.5}),
    ("start exclude: 0", {1: 0.5, 2: 0.5}),
    ("start exclude: 0 2", {1: 1.0}),
    ("start exclude: 0 right", {1: 1.0}),
])
def test_initial_distributions_and_wildcard_transitions(start, expected):
    model = pomdp_parser.parse(f"""
states: left middle right
actions: wait move
observations: hidden
{start}
T: * uniform
O: * uniform
""")
    assert model.start == pytest.approx(expected)
    assert model.transitions == {
        state: {action: {(0, 0): 1 / 3, (1, 0): 1 / 3, (2, 0): 1 / 3}
                for action in range(2)}
        for state in range(3)
    }


@pytest.mark.parametrize("selection", ["include", "exclude"])
def test_initial_state_indices_are_checked(selection):
    with pytest.raises(AssertionError):
        pomdp_parser.parse(f"""
states: left right
actions: wait
observations: seen
start {selection}: 2
T: wait identity
O: wait uniform
""")


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
    from scripts.convert_pps_to_pomdp import convert_pps_file, parse_pps_file, pps_to_pomdp

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
    converted = pps_to_pomdp(parse_pps_file(source))
    assert converted.transitions == model.transitions
    assert model.obs == ["seen", "hidden"]
    assert model.states == ["s0", "s1"]
    assert model.actions == ["step"]
    assert model.start == {0: 1.0}
    assert model.prio == {2: {0}, 1: {1}}
    positive_transitions = {
        state: {outcome: prob for outcome, prob in actions[0].items() if prob > 0}
        for state, actions in model.transitions.items()
    }
    assert positive_transitions == {
        0: {(0, 0): 0.5, (1, 1): 0.5},
        1: {(0, 0): 1.0},
    }


@pytest.mark.parametrize("priorities, expected", [
    ((0,), (0,)),
    ((0, 1, 2), (2, 1, 0)),
    ((1, 2, 3), (3, 2, 1)),
    ((0, 2, 5), (6, 4, 1)),
])
def test_pps_conversion_preserves_parity_acceptance(tmp_path, priorities, expected):
    from scripts.convert_pps_to_pomdp import convert_pps_file

    states = ",".join(f"{{{i};{priority};seen;T}}"
                      for i, priority in enumerate(priorities))
    transitions = "\n".join(f"TRANSITIONS: [{i},step,{{{i}}}] #"
                            for i in range(len(priorities)))
    source = tmp_path / "model.txt"
    source.write_text(f"""OBSERVATIONS: [seen] #
STATES: [{states}] #
ACTIONS: [step] #
{transitions}
""")
    output = tmp_path / "model.pomdp"
    assert convert_pps_file(source, output) == output
    model = pomdp_parser.parse(output.read_text())
    assert tuple(model.prioinv[i] for i in range(len(priorities))) == expected
    # Every possible set of recurring priorities must retain its acceptance.
    for size in range(1, len(priorities) + 1):
        for recurring in combinations(range(len(priorities)), size):
            assert min(priorities[i] for i in recurring) % 2 == max(
                model.prioinv[i] for i in recurring) % 2


def test_pps_cheese_maze_is_almost_sure_winning(tmp_path):
    from scripts.convert_pps_to_pomdp import convert_pps_file
    from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
    from pomdpy.belief_support_MDP import BeliefSuppMDP
    from pomdpy.revealing import is_strongly_revealing, make_strongly_revealing

    root = Path(__file__).resolve().parents[2]
    source = root / "examples/pps/Cheese maze 11 states/cheese_maze_problem_easy_spawn.txt"
    output = tmp_path / "cheese_maze.pomdp"
    assert convert_pps_file(source, output) == output
    model = make_strongly_revealing(pomdp_parser.parse(output.read_text()))
    assert is_strongly_revealing(model)
    mdp = BeliefSuppMDP(model)
    solver = ParityMDPSolver(mdp, verbose=False)
    winning, _, _ = solver.almostSureWin(max_priority=max(mdp.prio.values()))
    # Avoid states 8 and 9 while revisiting state 10: the minimum recurring
    # PPS priority is 2, even though priority 3 also recurs.
    assert mdp.start in winning


@pytest.mark.parametrize("recursive", [True, False])
def test_pps_directory_conversion_preserves_paths_and_continues_after_errors(
        tmp_path, recursive):
    from scripts import convert_pps_to_pomdp as converter

    source = tmp_path / "pps"
    (source / "nested").mkdir(parents=True)
    content = """OBSERVATIONS: [seen] #
STATES: [{0;0;seen;T}] #
ACTIONS: [step] #
TRANSITIONS: [0,step,{0}] #
"""
    files = {
        "same.txt": content,
        "nested/same.txt": content.replace("0;0;seen", "0;1;seen"),
        "broken.txt": content.replace("0;0;seen", "0;invalid;seen"),
        "notes.txt": "Not a PPS model.\n",
    }
    for name, text in files.items():
        (source / name).write_text(text)
    output = tmp_path / "converted"
    result = subprocess.run(
        [sys.executable, str(Path(converter.__file__).resolve()),
         "--dir", str(source), "--output", str(output),
         *([] if recursive else ["--no-recursive"])],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected_files = {"same.pomdp"}
    if recursive:
        expected_files.add("nested/same.pomdp")
    assert {p.relative_to(output).as_posix() for p in output.rglob("*.pomdp")} == expected_files
    assert f"Success: {len(expected_files)}" in result.stdout
    assert "Failed: 1" in result.stdout
    for name in expected_files:
        model = pomdp_parser.parse((output / name).read_text())
        assert model.prioinv == {0: 1 if name.startswith("nested/") else 0}
        assert model.start == {0: 1.0}
        assert model.transitions == {0: {0: {(0, 0): 1.0}}}
    assert all((source / name).read_text() == text for name, text in files.items())
