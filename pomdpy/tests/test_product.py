"""Regression tests for the observation-synchronous product."""

from collections import Counter

import pytest
import spot

from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
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


def test_product_export_requires_factor_tables(labelled_product, tmp_path):
    _, _, product = labelled_product
    with pytest.raises(ValueError, match="export requires transition.*observation"):
        product.to_pomdp_string()

    path = tmp_path / "product.pomdp"
    path.write_text("previous contents")
    with pytest.raises(ValueError, match="export requires transition.*observation"):
        product.to_pomdp_file(path)
    assert path.read_text() == "previous contents"
    assert list(tmp_path.iterdir()) == [path]


def test_product_joint_transitions_survive_compute_calls(labelled_product):
    _, _, product = labelled_product
    transitions = product.transitions
    product.computeTrans()
    with pytest.raises(ValueError, match="requires both T and O"):
        product.computeTrans(force=True)
    assert product.transitions is transitions
    assert is_strongly_revealing(product)


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


def test_product_omits_zero_outcomes(labelled_product, monkeypatch):
    env, aut, original = labelled_product
    env.setObs([*env.obs, "never"])
    for state in range(len(env.states)):
        for action in range(len(env.actions)):
            for dst in range(len(env.states)):
                env.T[state][action].setdefault(dst, 0.0)
            for obs in range(len(env.obs)):
                env.O[action][state].setdefault(obs, 0.0)

    from pomdpy import product as product_module
    next_state = product_module.get_next_aut_state_by_observation

    def check_observation(env, aut, obs, state, atoms):
        assert obs != env.obsinv["never"]
        return next_state(env, aut, obs, state, atoms)

    monkeypatch.setattr(product_module, "get_next_aut_state_by_observation", check_observation)
    rebuilt = ProductPOMDP(env, aut)
    assert rebuilt.transitions == original.transitions
    assert all(p > 0 for actions in rebuilt.transitions.values()
               for row in actions.values() for p in row.values())


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


@pytest.mark.parametrize("unused_atoms", [False, True])
def test_automaton_update_decodes_product_observations(labelled_product, unused_atoms):
    env, aut, product = labelled_product
    if unused_atoms:
        # These labels do not occur in the automaton and must not affect updates.
        env.addAtom(7, ["common", "wait"])
    for q in range(aut.num_states()):
        for base_obs in range(len(env.obs)):
            q_prime = 1 - q if base_obs in env.atoms[0] else q
            obs = base_obs * aut.num_states() + q_prime
            assert get_next_aut_state_by_observation(product, aut, obs, q) == q_prime


@pytest.mark.parametrize("options", [("BA",), ("parity max even", "colored")])
def test_automaton_priorities_are_read_once_per_state(labelled_product, monkeypatch, options):
    env, _, _ = labelled_product
    aut = spot.split_edges(spot.translate(
        "GFp0", *options, "deterministic", "complete", "SBAcc"
    ))
    state_acc_sets = aut.state_acc_sets
    calls = Counter()

    def count_queries(q):
        calls[q] += 1
        return state_acc_sets(q)

    monkeypatch.setattr(aut, "state_acc_sets", count_queries)
    for _ in range(2):
        calls.clear()
        product = ProductPOMDP(env, aut)
        assert calls == dict.fromkeys(range(aut.num_states()), 1)
        assert set(product.prioinv.values()) == {1, 2}

        calls.clear()
        mdp = BeliefSuppMDP(product, aut)
        visited = {q for support in mdp.states for _, q in support}
        assert calls == dict.fromkeys(visited, 1)
        assert set(mdp.prio.values()) == {1, 2}
        for state, support in enumerate(mdp.states):
            assert mdp.prio[state] == max(
                product.prioinv[q * len(env.states) + s] for s, q in support
            )


def test_rebuilding_product_uses_current_observation_labels(labelled_product):
    env, aut, original = labelled_product
    # Reuse the same objects after changing their labels: cached successors
    # from the first construction must not affect the second product.
    env.atoms[0] = set(range(len(env.obs))) - env.atoms[0]
    rebuilt = ProductPOMDP(env, aut)
    for source, actions in original.transitions.items():
        for action, row in actions.items():
            expected = {}
            for (destination, obs), probability in row.items():
                old_q, dst = divmod(destination, len(env.states))
                # Flipping p0 flips the successor in the two-state toggle automaton.
                new_q = 1 - old_q
                new_obs = (obs // aut.num_states()) * aut.num_states() + new_q
                expected[new_q * len(env.states) + dst, new_obs] = probability
            assert rebuilt.transitions[source][action] == expected


@pytest.mark.parametrize("formula, expected_winning", [("1", True), ("0", False)])
def test_constant_objectives(labelled_product, formula, expected_winning):
    env, _, _ = labelled_product
    aut = spot.translate(
        formula, "parity max even", "deterministic", "complete", "SBAcc", "colored"
    )
    aut = spot.split_edges(aut)
    mdp = BeliefSuppMDP(ProductPOMDP(env, aut), aut)
    solver = ParityMDPSolver(mdp, verbose=False)
    winning, _, _ = solver.almostSureWin(max_priority=max(mdp.prio.values()))
    assert (mdp.start in winning) == expected_winning
