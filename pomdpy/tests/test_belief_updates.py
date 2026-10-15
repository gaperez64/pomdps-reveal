"""Simulation, belief updates, and support abstraction on small models."""

import numpy as np
import pytest

from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.pomdp import POMDP
from pomdpy.sim import update_belief


@pytest.mark.parametrize("observation, expected", [
    ("seen", {0: 5 / 17, 1: 12 / 17}),
    ("impossible", {}),
])
def test_belief_update_conditions_on_the_observation(observation, expected):
    env = pomdp_parser.parse("""
states: left right absent
actions: step
observations: seen hidden impossible
start: 0.75 0.25 0
T: step
0.5 0.5 0
1 0 0
0 0 1
O: step
0.2 0.8 0
0.8 0.2 0
0 1 0
""")
    belief = dict(env.start)
    # Before observation: (5/8, 3/8, 0). Observing 'seen' gives
    # unnormalized masses (1/8, 3/10, 0), hence (5/17, 12/17, 0).
    posterior = update_belief(env, belief, 0, env.obsinv[observation])
    assert posterior == pytest.approx(expected)
    assert belief == env.start


def test_support_update_ignores_zero_mass_and_merges_equal_successors():
    env = pomdp_parser.parse("""
states: left right absent
actions: stay
observations: common other left right absent
prio 1: left
prio 2: right
prio 3: absent
start: 0.75 0.25 0
T: stay identity
O: stay
0.25 0.25 0.5 0 0
0.25 0.25 0 0.5 0
0 0 0 0 1
""")
    mdp = BeliefSuppMDP(env)
    assert mdp.states[mdp.start] == (0, 1)
    assert set(mdp.states) == {(0, 1), (0,), (1,)}
    # Two observations give the same ambiguous support; it appears only once.
    successors = mdp.trans[mdp.start][0]
    assert len(successors) == 3
    assert {mdp.states[i] for i in successors} == {(0, 1), (0,), (1,)}
    for state, support in enumerate(mdp.states):
        assert mdp.prio[state] == (2 if 1 in support else 1)
        if len(support) == 1:
            assert mdp.trans[state][0] == [state]


@pytest.mark.parametrize("choice, state, observation", [
    (0, 1, "seen-left"), (2, 0, "seen-right"),
])
def test_step_keeps_the_sampled_state_and_observation_together(
    monkeypatch, choice, state, observation
):
    env = POMDP()
    env.setStates(["left", "right", "initial"])
    env.setActions(["wait", "move"])
    env.setObs(["seen-left", "seen-right", "never"])
    env.curstate = 2
    env.transitions = {2: {
        0: {(2, 2): 1.0},
        1: {(1, 0): 0.75, (2, 2): 0.0, (0, 1): 0.25},
    }}

    def choose_outcome(size, p):
        assert size == 3
        assert p == pytest.approx([0.75, 0.0, 0.25])
        return choice

    monkeypatch.setattr(np.random, "choice", choose_outcome)
    assert env.step(env.actionsinv["move"]) == observation
    assert env.curstate == state
