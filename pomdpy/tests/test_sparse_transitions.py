"""Sparse joint transitions preserve probabilities and the T/O file format."""

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
    validate_transition_rows(loaded)


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
def test_sparse_transitions_do_not_hide_invalid_nonzero_probabilities(probability):
    model = POMDP()
    model.setStates(["state"])
    model.setActions(["action"])
    model.setObs(["observation"])
    model.T = {0: {0: {0: probability}}}
    model.O = {0: {0: {0: 1.0}}}
    model.computeTrans()
    assert (0, 0) in model.transitions[0][0]
    with pytest.raises(ValueError, match="Invalid transition probabilities"):
        validate_transition_rows(model)
