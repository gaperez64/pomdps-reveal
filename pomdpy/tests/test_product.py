"""Regression tests for the observation-synchronous product."""

import pytest
import spot

from pomdpy.belief_support_MDP import BeliefSuppMDP, get_next_aut_state_by_observation
from pomdpy.pomdp import AtomicPropPOMDP
from pomdpy.product import ProductPOMDP
from pomdpy.revealing import is_strongly_revealing


@pytest.fixture
def labelled_product():
    env = AtomicPropPOMDP()
    env.setStates(["left", "right"])
    env.setActions(["mix", "stay"])
    env.setObs(["common", "left", "right", "wait", "wait-left", "wait-right"])
    env.setUniformStart()
    env.addUniformTrans(act=0)
    env.addIdentityTrans(act=1)
    for action in range(2):
        env.O[action] = {
            0: {0: 0.25, 1: 0.25, 3: 0.25, 4: 0.25},
            1: {0: 0.25, 2: 0.25, 3: 0.25, 5: 0.25},
        }
    env.addAtom(0, ["common", "left", "right"])
    env.computeTrans()
    # p0 toggles the automaton state; its absence leaves the state unchanged.
    aut = spot.automaton('''HOA: v1
States: 2
Start: 0
AP: 1 "p0"
acc-name: parity max even 2
Acceptance: 2 Inf(0) & Fin(1)
properties: trans-labels explicit-labels state-acc complete deterministic
--BODY--
State: 0 {0}
[0] 1
[!0] 0
State: 1 {1}
[0] 0
[!0] 1
--END--
''')
    return env, aut, ProductPOMDP(env, aut)


def test_product_signals_record_the_next_automaton_state(labelled_product):
    env, aut, product = labelled_product
    assert is_strongly_revealing(env, preserve_labels=True)
    assert is_strongly_revealing(product)
    assert len(product.obs) == len(env.obs) * aut.num_states()
    assert product.start == {0: 0.5, 1: 0.5}
    for state, actions in product.transitions.items():
        source, q = state % len(env.states), state // len(env.states)
        for action, row in actions.items():
            assert sum(row.values()) == pytest.approx(1.0)
            projected = {}
            for (destination, obs), probability in row.items():
                if probability <= 0:
                    continue
                dst = destination % len(env.states)
                q_prime = destination // len(env.states)
                base_obs, observed_q = divmod(obs, aut.num_states())
                assert observed_q == q_prime
                assert q_prime == (1 - q if base_obs in env.atoms[0] else q)
                assert product.obs[obs] == f"{env.obs[base_obs]}-{q_prime}"
                assert (dst, base_obs) not in projected
                projected[dst, base_obs] = probability
            assert projected == {
                key: p for key, p in env.transitions[source][action].items() if p > 0
            }


def test_added_signal_component_preserves_reachable_belief_supports(labelled_product):
    env, aut, product = labelled_product
    # Recover the old product by forgetting the signal's automaton component.
    old_product = ProductPOMDP(env, aut)
    old_product.obs = list(env.obs)
    old_product.obsinv = dict(env.obsinv)
    old_product.transitions = {}
    for source, actions in product.transitions.items():
        old_product.transitions[source] = {}
        for action, row in actions.items():
            old_row = {}
            for (destination, obs), probability in row.items():
                key = (destination, obs // aut.num_states())
                old_row[key] = old_row.get(key, 0.0) + probability
            old_product.transitions[source][action] = old_row
    assert not is_strongly_revealing(old_product)
    old_mdp = BeliefSuppMDP(old_product, aut)
    new_mdp = BeliefSuppMDP(product, aut)
    assert new_mdp.states == old_mdp.states
    assert new_mdp.trans == old_mdp.trans
    assert new_mdp.prio == old_mdp.prio


def test_automaton_update_decodes_product_observations(labelled_product):
    env, aut, product = labelled_product
    for q in range(aut.num_states()):
        for base_obs in range(len(env.obs)):
            q_prime = 1 - q if base_obs in env.atoms[0] else q
            obs = base_obs * aut.num_states() + q_prime
            assert get_next_aut_state_by_observation(product, aut, obs, q) == q_prime
