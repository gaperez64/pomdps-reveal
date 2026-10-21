"""Sparse probability tables preserve probabilities and the T/O file format."""

from copy import deepcopy

import pytest

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.pomdp import POMDP, AtomicPropPOMDP, ParityPOMDP
from pomdpy.revealing import validate_transition_rows


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
