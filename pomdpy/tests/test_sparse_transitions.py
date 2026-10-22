"""Sparse probability tables preserve probabilities and the T/O file format."""

from copy import deepcopy

import pytest

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.pomdp import POMDP, AtomicPropPOMDP, ParityPOMDP
from pomdpy.revealing import validate_transition_rows


@pytest.mark.parametrize("model_type, header, specification", [
    (POMDP, "# POMDP file\n", ""),
    (ParityPOMDP, "# POMDP file converted from PPS format\n\n",
     "prio 2: left\nprio 12: right\n\n"),
    (AtomicPropPOMDP, "# POMDP file with atomic propositions\n",
     "atom 2: seen hidden\natom 12: hidden\n\n"),
    (ParityPOMDP, "# POMDP file converted from PPS format\n\n", ""),
    (AtomicPropPOMDP, "# POMDP file with atomic propositions\n", ""),
])
def test_export_preserves_format_and_streams_to_file(tmp_path, monkeypatch,
                                                    model_type, header, specification):
    model = model_type()
    model.setStates(["right", "left"])
    model.setActions(["mix", "missing"])
    model.setObs(["seen", "hidden"])
    model.start = {0: 1 / 3, 1: 2 / 3}
    model.T = {0: {0: {0: 0.5, 1: 0.5}}}
    model.O = {0: {0: {0: 1e-300, 1: 1.0}}}
    if specification:
        if isinstance(model, ParityPOMDP):
            model.addPriority(12, ["right"])
            model.addPriority(2, ["left"])
        else:
            model.addAtom(12, ["hidden"])
            model.addAtom(2, ["hidden", "seen"])

    expected = header + """states: right left
actions: mix missing
observations: seen hidden

""" + specification + """start: 0.33333333333333 0.66666666666667

T:mix
0.5 0.5
0 0

T:missing
0 0
0 0

O:mix
1e-300 1
0 0

O:missing
0 0
0 0
"""
    assert model.to_pomdp_string() == expected

    def buffer_full_output():
        pytest.fail("File export must not build the full output string")

    monkeypatch.setattr(model, "to_pomdp_string", buffer_full_output)
    path = tmp_path / "export.pomdp"
    path.write_text("previous contents")
    path.chmod(0o640)
    mode = path.stat().st_mode
    model.to_pomdp_file(path)
    assert path.read_bytes() == expected.encode()
    assert path.stat().st_mode == mode
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error_type", [TimeoutError, KeyboardInterrupt])
def test_interrupted_export_preserves_destination(tmp_path, monkeypatch, existing,
                                                  error_type):
    model = POMDP()
    path = tmp_path / "export.pomdp"
    if existing:
        path.write_text("previous contents")

    def interrupted_lines():
        yield "# partial output"
        if existing:
            assert path.read_text() == "previous contents"
        else:
            assert not path.exists()
        raise error_type("interrupted export")

    monkeypatch.setattr(model, "_pomdp_lines", interrupted_lines)
    with pytest.raises(error_type, match="interrupted export"):
        model.to_pomdp_file(path)
    if existing:
        assert path.read_text() == "previous contents"
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


def test_export_formats_sparse_rows_in_index_order():
    model = POMDP()
    model.setStates(4)
    model.setActions(["move"])
    model.setObs(4)
    row = {3: 1e-300, 2: -0.0, 0: 1 / 3, -1: 0.9, 4: 0.9}
    original = row.copy()
    model.start = row
    model.T = {0: {0: row}}
    model.O = {0: {0: row}}

    exported = model.to_pomdp_string()
    formatted = "0.33333333333333 0 -0 1e-300"
    assert f"start: {formatted}\n" in exported
    for header in ("T:move", "O:move"):
        assert f"{header}\n{formatted}\n" + "0 0 0 0\n" * 3 in exported
    assert row == original


@pytest.mark.parametrize("model_type", [POMDP, AtomicPropPOMDP, ParityPOMDP])
def test_sparse_transitions_preserve_factors_and_roundtrip(model_type):
    model = model_type()
    model.setStates(["left", "right"])
    model.setActions(["mix", "stay"])
    model.setObs(["seen", "hidden", "right", "never"])
    model.setUniformStart()
    model.addTrans([0.5, 0.5, 1.0, 0.0], act=0)
    model.addIdentityTrans(act=1)
    model.addObs([0.25, 0.75, 0.0, 0.0, 1e-300, 0.0, 1.0, 0.0])
    assert model.T == {0: {0: {0: 0.5, 1: 0.5}, 1: {0: 1.0}},
                       1: {0: {0: 1.0}, 1: {1: 1.0}}}
    assert model.O == {action: {0: {0: 0.25, 1: 0.75}, 1: {0: 1e-300, 2: 1.0}}
                       for action in range(2)}
    if isinstance(model, AtomicPropPOMDP):
        model.addAtom(0, ["right"])
    if isinstance(model, ParityPOMDP):
        model.addPriority(1, ["left"])
        model.addPriority(2, ["right"])
    factors = deepcopy((model.T, model.O))
    exported = model.to_pomdp_string()

    model.computeTrans()
    assert model.transitions == {
        0: {0: {(0, 0): 0.125, (0, 1): 0.375, (1, 0): 5e-301, (1, 2): 0.5},
            1: {(0, 0): 0.25, (0, 1): 0.75}},
        1: {0: {(0, 0): 0.25, (0, 1): 0.75},
            1: {(1, 0): 1e-300, (1, 2): 1.0}},
    }
    assert (model.T, model.O) == factors
    assert model.to_pomdp_string() == exported
    loaded = pomdp_parser.parse(exported)
    assert type(loaded) is model_type
    assert loaded.transitions == model.transitions
    assert loaded.start == model.start
    assert (loaded.T, loaded.O) == factors
    validate_transition_rows(loaded)


def test_zero_assignments_override_existing_factor_entries():
    model = pomdp_parser.parse("""
states: left middle right
actions: act spare
observations: seen hidden other
start: uniform
T: * uniform
O: * uniform
T: act
0 1 0
0 0 0
1e-300 0 1
O: act
0 1 0
0 0 0
1e-300 0 1
""")
    expected = {0: {1: 1.0}, 1: {}, 2: {0: 1e-300, 2: 1.0}}
    assert {state: model.T[state][0] for state in range(3)} == expected
    assert model.O[0] == expected
    uniform = {state: {i: 1 / 3 for i in range(3)} for state in range(3)}
    assert {state: model.T[state][1] for state in range(3)} == uniform
    assert model.O[1] == uniform

    # A row update must remove old positive entries without affecting other rows.
    model.addTrans([0.5, 0.5, 0.0], act=0, src=2)
    model.addObs([0.5, 0.5, 0.0], act=0, dst=2)
    expected[2] = {0: 0.5, 1: 0.5}
    assert {state: model.T[state][0] for state in range(3)} == expected
    assert model.O[0] == expected


def test_identity_replaces_existing_transitions_for_the_selected_action():
    model = pomdp_parser.parse("""
states: left right
actions: move spare
observations: seen
start: uniform
T: * uniform
T: move identity
O: * uniform
""")
    assert {state: model.T[state][0] for state in range(2)} == {
        0: {0: 1.0}, 1: {1: 1.0},
    }
    assert {state: model.T[state][1] for state in range(2)} == {
        0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5},
    }
    validate_transition_rows(model)
    assert pomdp_parser.parse(model.to_pomdp_string()).transitions == model.transitions


def test_rebuild_after_editing_transition_and_observation_tables():
    model = pomdp_parser.parse("""
states: left right
actions: wait
observations: left right
prio 0: left
prio 1: right
start include: left
T: wait uniform
O: wait
1 0
0 1
""")
    mdp = BeliefSuppMDP(model)
    assert mdp.start not in ParityMDPSolver(mdp).almostSureWin()[0]

    model.addIdentityTrans()
    model.computeTrans(force=True)
    assert model.transitions == {0: {0: {(0, 0): 1.0}}, 1: {0: {(1, 1): 1.0}}}
    assert pomdp_parser.parse(model.to_pomdp_string()).transitions == model.transitions
    mdp = BeliefSuppMDP(model)
    assert mdp.start in ParityMDPSolver(mdp).almostSureWin()[0]

    model.addObs([0, 1, 1, 0])
    model.computeTrans(force=True)
    assert model.transitions == {0: {0: {(0, 1): 1.0}}, 1: {0: {(1, 0): 1.0}}}
    assert pomdp_parser.parse(model.to_pomdp_string()).transitions == model.transitions


@pytest.mark.parametrize("selection", [{"inc": ["left"]}, {"exc": ["right"]}])
def test_uniform_start_replaces_the_previous_distribution(selection):
    model = POMDP()
    model.setStates(["left", "right"])
    model.setUniformStart()
    model.setUniformStart(**selection)
    assert model.start == {0: 1.0}
    model.setUniformStart()
    assert model.start == {0: 0.5, 1: 0.5}


@pytest.mark.parametrize("ids", [False, True])
def test_priority_reassignment_preserves_consistent_export(ids):
    model = pomdp_parser.parse("""
states: left right
actions: stay
observations: seen
prio 2: left right
start: uniform
T: stay identity
O: stay uniform
""")
    states = [0, 1] if ids else ["left", "right"]
    model.addPriority(0, states[:1], ids=ids)
    model.addPriority(0, states[:1], ids=ids)  # Repeating an assignment is harmless.
    assert model.prio == {0: {0}, 2: {1}}
    model.addPriority(1, states[1:], ids=ids)
    assert model.prio == {0: {0}, 1: {1}}
    model.addPriority(1, states[:1], ids=ids)
    assert model.prio == {1: {0, 1}}
    assert model.prioinv == {0: 1, 1: 1}
    model.addPriority(0, model.prio[1], ids=True)
    assert model.prio == {0: {0, 1}}
    assert model.prioinv == {0: 0, 1: 0}
    loaded = pomdp_parser.parse(model.to_pomdp_string())
    assert loaded.prio == model.prio
    assert loaded.prioinv == model.prioinv


def test_factor_rows_can_be_filled_cleared_and_reused_in_place():
    model = POMDP()
    model.setStates(["left", "middle", "right"])
    model.setActions(["move"])
    model.setObs(["seen", "hidden", "other"])
    model.addTrans([0.0] * 9)
    model.addObs([0.0] * 9)
    assert model.T == {state: {0: {}} for state in range(3)}
    assert model.O == {0: {state: {} for state in range(3)}}
    trans_row, obs_row = model.T[0][0], model.O[0][0]

    for values, expected in [
        ([0.0, 0.5, 0.5], [(1, 0.5), (2, 0.5)]),
        ([0.25, 0.0, 0.75], [(2, 0.75), (0, 0.25)]),
        ([-0.0, 0.0, 0.0], []),
        ([1e-300, 0.0, 1.0], [(0, 1e-300), (2, 1.0)]),
    ]:
        model.addTrans(values, act=0, src=0)
        model.addObs(values, act=0, dst=0)
        assert model.T[0][0] is trans_row
        assert model.O[0][0] is obs_row
        assert list(trans_row.items()) == expected
        assert list(obs_row.items()) == expected
    assert model.T[1][0] == model.T[2][0] == {}
    assert model.O[0][1] == model.O[0][2] == {}


def test_sparse_transitions_keep_empty_rows_and_existing_joint_tables():
    model = POMDP()
    model.setStates(["left", "right", "absent"])
    model.setActions(["move", "missing"])
    model.setObs(["seen", "hidden"])
    model.T = {0: {0: {0: 0.5, 1: 0.5, 2: 0.0}}, 1: {0: {1: 1.0}}}
    model.O = {0: {0: {0: 1.0, 1: 0.0}}}
    model.computeTrans()
    assert model.transitions == {
        0: {0: {(0, 0): 0.5}, 1: {}},
        1: {0: {}, 1: {}},
        2: {0: {}, 1: {}},
    }
    with pytest.raises(ValueError, match="sum to 0.5, expected 1"):
        validate_transition_rows(model)

    # Explicit joint tables remain authoritative, including user-supplied zeros.
    model.transitions = {0: {0: {(0, 0): 0.0, (1, 1): 1.0}}}
    existing = deepcopy(model.transitions)
    model.computeTrans()
    assert model.transitions == existing


@pytest.mark.parametrize("probability", [-0.25, float("inf"), float("nan")])
@pytest.mark.parametrize("factor", ["T", "O"])
def test_sparse_transitions_do_not_hide_invalid_nonzero_probabilities(probability, factor):
    model = POMDP()
    model.setStates(["state"])
    model.setActions(["action"])
    model.setObs(["observation"])
    model.addTrans([probability if factor == "T" else 1.0])
    model.addObs([probability if factor == "O" else 1.0])
    model.computeTrans()
    assert (0, 0) in model.transitions[0][0]
    with pytest.raises(ValueError, match="Invalid transition probabilities"):
        validate_transition_rows(model)
