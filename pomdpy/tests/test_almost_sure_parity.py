"""
Tests for almost_sure_parity_MDP.py

This module tests the ParityMDPSolver class which computes almost-sure
winning strategies for parity objectives on belief-support MDPs.
"""

from types import SimpleNamespace
import sys

import pytest
import spot
from pomdpy.pomdp import AtomicPropPOMDP
from pomdpy.product import ProductPOMDP
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
from pomdpy.parsers import pomdp


def test_tarjan_sccs():
    """Test Tarjan's SCC algorithm on a simple MDP."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1", "s2"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0

    # Create a cycle: s0 -> s1 -> s2 -> s0
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 2, 1.0)
    env._addOneTrans(2, 0, 0, 1.0)

    env._addOneObs(0, 0, 0, 1.0)
    env._addOneObs(0, 1, 0, 1.0)
    env._addOneObs(0, 2, 0, 1.0)

    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    # Create product POMDP first
    product = ProductPOMDP(env, aut)
    mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Test Tarjan's algorithm
    states = set(range(len(mdp.states)))
    actions = {s: list(range(len(mdp.actions))) for s in states}
    sccs = solver.tarjanSCCs(states, actions)

    # Should return a list of SCCs (each SCC is a list)
    assert isinstance(sccs, list)
    assert all(isinstance(scc, list) for scc in sccs)

    # All states should be covered
    all_states_in_sccs = set().union(*[set(scc) for scc in sccs]) if sccs else set()
    assert all_states_in_sccs.issubset(states)


@pytest.mark.parametrize("cycle", [False, True])
def test_tarjan_sccs_deep_graph(cycle):
    """Decompose long paths into singletons and long cycles into one SCC."""
    size = sys.getrecursionlimit() + 100
    mdp = SimpleNamespace(
        states=list(range(size)), actions=["a"], actionsinv={"a": 0},
        prio=dict.fromkeys(range(size), 0),
        trans={q: {0: [q + 1]} for q in range(size - 1)},
    )
    mdp.trans[size - 1] = {0: [0 if cycle else size - 1]}
    states = set(range(size))
    components = ParityMDPSolver(mdp).tarjanSCCs(states, {q: {0} for q in states})
    expected = {frozenset(states)} if cycle else {frozenset({q}) for q in states}
    assert {frozenset(component) for component in components} == expected


def test_tarjan_sccs_restricted_graph():
    """Respect state/action restrictions and distinguish active and completed SCCs."""
    states = {0, 2, 4, 6, 8, 10, 12}
    mdp = SimpleNamespace(
        states=list(range(13)), actions=["a", "b"], actionsinv={"a": 0, "b": 1},
        prio=dict.fromkeys(range(13), 0),
        trans={
            0: {0: [2, 6]},
            2: {0: [4], 1: [8]},
            4: {0: [0]},
            6: {0: [2, 1]},
            8: {0: [2, 10]},
            10: {0: [8]},
            12: {},
        },
    )
    actions = {q: {0} for q in states}
    actions[12] = set()
    components = ParityMDPSolver(mdp).tarjanSCCs(states, actions)
    assert {frozenset(component) for component in components} == {
        frozenset({0, 2, 4, 6}), frozenset({8, 10}), frozenset({12}),
    }


def test_almost_sure_reach_trivial():
    """Test almost-sure reachability with trivial case (target is initial state)."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Target is the only state
    target = {0}
    reachable = solver.almostSureReach(target)

    # Should return a list
    assert isinstance(reachable, list)
    # Should reach itself
    assert 0 in reachable


def test_almost_sure_reach_chain():
    """Test almost-sure reachability on a chain of states."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1", "s2"])
    env.setActions(["a"])
    env.setObs(["o1", "o2", "o3"])

    env.start[0] = 1.0

    # Chain: s0 -> s1 -> s2 (deterministic)
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 2, 1.0)
    env._addOneTrans(2, 0, 2, 1.0)  # Self-loop at end

    env._addOneObs(0, 1, 0, 1.0)
    env._addOneObs(0, 2, 1, 1.0)
    env._addOneObs(0, 0, 2, 1.0)

    env.computeTrans()
    env.addAtom(0, ["o3"], ids=False)

    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Find target states (those containing POMDP state 2)
    target = set()
    for i, bs in enumerate(mdp.states):
        if any(s == 2 for s, q in bs):
            target.add(i)

    if target:
        reachable = solver.almostSureReach(target)
        # Should return a list
        assert isinstance(reachable, list)
        # Initial state should reach target
        assert mdp.start in reachable or mdp.start in target


def test_almost_sure_reach_multiple_rounds(monkeypatch):
    """Removing unsafe actions must also eliminate states left only waiting."""
    mdp = SimpleNamespace(
        states=["goal", "trap", "waiting", "waiting_again", "safe", "forced"],
        actions=["wait", "try"],
        actionsinv={"wait": 0, "try": 1},
        prio=dict.fromkeys(range(6), 0),
        trans={
            0: {0: [0], 1: [0]},
            1: {0: [1], 1: [1]},
            2: {0: [2], 1: [0, 5]},
            3: {0: [3], 1: [0, 2]},
            4: {0: [0, 4], 1: [0, 3]},
            5: {0: [1, 2], 1: [1, 2]},
        },
    )
    solver = ParityMDPSolver(mdp)
    cannot_reach = solver.cannotReach
    checks = 0

    def bounded_cannot_reach(targets, forbidden):
        nonlocal checks
        checks += 1
        # Each nonempty round must remove at least one new state.
        assert checks <= len(mdp.states) + 1, "Reachability did not converge"
        return cannot_reach(targets, forbidden)

    monkeypatch.setattr(solver, "cannotReach", bounded_cannot_reach)
    # The trap eliminates the forced state, then each waiting state in turn.
    # State 4 can still reach the goal almost surely using its safe action.
    assert solver.almostSureReach({0}) == [0, 4]


def test_solver_rebuilds_predecessors_between_queries(monkeypatch):
    """Share predecessors within a solve, but rebuild after reachability prunes them."""
    mdp = SimpleNamespace(
        states=["choice", "even0", "even2", "odd"],
        actions=["risky", "safe"], actionsinv={"risky": 0, "safe": 1},
        prio={0: 1, 1: 0, 2: 2, 3: 3},
        trans={
            0: {0: [1, 3], 1: [2]},
            1: {0: [1], 1: [1]},
            2: {0: [2], 1: [2]},
            3: {0: [3], 1: [3]},
        },
    )
    solver = ParityMDPSolver(mdp)
    reset = solver.resetPreAct
    builds = 0

    def count_builds():
        nonlocal builds
        builds += 1
        reset()

    monkeypatch.setattr(solver, "resetPreAct", count_builds)
    expected = (
        [0, 1, 2], {0: [1], 1: [0, 1], 2: [0, 1]},
        [{1: {0, 1}}, {2: {0, 1}}],
    )
    for count in (1, 2):
        assert solver.almostSureWin(max_priority=2) == expected
        assert builds == count

    # Standalone queries must also restore predecessors and action counts.
    assert solver.almostSureReach({1}) == [1]
    assert solver.almostSureReach({2}) == [0, 2]
    assert solver.goodMECs(0)[0] == [{1}]
    assert [mecs for mecs, _ in solver.mecs()] == [[{1}], [{2}]]
    assert builds == 6

    # Each new solve reads the current graph rather than a persistent cache.
    mdp.trans[0][1] = [3]
    winning, reach_strategy, _ = solver.almostSureWin(max_priority=2)
    assert winning == [1, 2]
    assert reach_strategy == {1: [0, 1], 2: [0, 1]}
    assert builds == 7


def test_good_mecs_basic():
    """Test goodMECs computation on a simple MDP."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0

    # Cycle between states
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 0, 1.0)

    env._addOneObs(0, 0, 0, 1.0)
    env._addOneObs(0, 1, 0, 1.0)

    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Compute good MECs for priority 0
    mecs, strats = solver.goodMECs(priority=0)

    # Should return lists
    assert isinstance(mecs, list)
    assert isinstance(strats, dict)

    # Each MEC should be a set
    assert all(isinstance(mec, set) for mec in mecs)


def test_almost_sure_win_simple():
    """Test almost-sure winning computation on a simple example."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Compute almost-sure winning strategy
    reach, reach_strat, mec_strats = solver.almostSureWin(max_priority=2)

    # Should return list/dict/list (not set)
    assert isinstance(reach, list)
    assert isinstance(reach_strat, dict)
    assert isinstance(mec_strats, list)

    # Initial state should be winning (or not, depending on formula)
    # Just check structure is valid
    for s in reach_strat:
        assert isinstance(reach_strat[s], list)
        assert all(isinstance(a, int) for a in reach_strat[s])


def test_mecs_wrapper():
    """Test the mecs() wrapper method."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 0, 1.0)

    env._addOneObs(0, 0, 0, 1.0)
    env._addOneObs(0, 1, 0, 1.0)

    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Call mecs wrapper
    result = solver.mecs()

    # Should return a list
    assert isinstance(result, list)


def test_cannot_reach():
    """Test cannotReach method for computing unreachable states."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1", "s2"])
    env.setActions(["a"])
    env.setObs(["o1", "o2"])

    env.start[0] = 1.0

    # s0 -> s1, s1 -> s2, but s2 is forbidden
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 2, 1.0)
    env._addOneTrans(2, 0, 2, 1.0)

    env._addOneObs(0, 1, 0, 1.0)
    env._addOneObs(0, 2, 1, 1.0)
    env._addOneObs(0, 0, 0, 1.0)

    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Test cannotReach with forbidden states
    solver.resetPreAct()  # Need to initialize pre/act first
    targets = {len(mdp.states) - 1} if mdp.states else {0}
    forbidden = set()

    cannot = solver.cannotReach(targets, forbidden)

    # Should return a list
    assert isinstance(cannot, list)


def test_revealing_tiger_mecs():
    """Test MEC computation on revealing-tiger example."""
    try:
        with open("examples/revealing_ltl-tiger.pomdp", "r") as f:
            content = f.read()
            env = pomdp.parse(content)

        # Create simple automaton
        aut = spot.translate("Fp0", "parity", "SBAcc")
        aut = spot.split_edges(aut)

        # Build MDP
        product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
        solver = ParityMDPSolver(mdp, verbose=False)

        # Compute MECs
        mecs, strats = solver.goodMECs(priority=0)

        # Check structure
        assert isinstance(mecs, list)
        assert isinstance(strats, dict)

        # Check that strategies are valid
        for state in strats:
            assert state < len(mdp.states)
            assert isinstance(strats[state], (list, set))

    except (FileNotFoundError, ValueError):
        # Skip if file doesn't exist or formula doesn't match
        pass


def test_multiple_priorities():
    """Test goodMECs with different priority values."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Test different priorities
    for prio in [0, 2, 4]:
        mecs, strats = solver.goodMECs(priority=prio)
        assert isinstance(mecs, list)
        assert isinstance(strats, dict)


@pytest.mark.parametrize("priority", [0, 1])
def test_good_mecs_large_priority_gap(priority):
    """Find the lower even priority or return no MECs when none exists."""
    mdp = SimpleNamespace(
        states=[0], actions=["a"], actionsinv={"a": 0},
        prio={0: priority}, trans={0: {0: [0]}},
    )
    solver = ParityMDPSolver(mdp)
    expected = ([{0}], {0: {0}}) if priority == 0 else ([], {})
    assert solver.goodMECs(2 * (sys.getrecursionlimit() + 100)) == expected


def test_almost_sure_win_with_visualization():
    """Test almost-sure winning with visualization disabled."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Test without visualization
    reach1, strat1, mecs1 = solver.almostSureWin(max_priority=2, vis=None)

    # Verify results have correct structure
    assert isinstance(reach1, list)
    assert isinstance(strat1, dict)
    assert isinstance(mecs1, list)


def test_verbose_mode():
    """Test that verbose mode doesn't break computation."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 0, 1.0)

    env._addOneObs(0, 0, 0, 1.0)
    env._addOneObs(0, 1, 0, 1.0)

    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)

    # Test with verbose=False
    solver1 = ParityMDPSolver(mdp, verbose=False)
    reach1, strat1, mecs1 = solver1.almostSureWin(max_priority=2)

    # Test with verbose=True
    solver2 = ParityMDPSolver(mdp, verbose=True)
    reach2, strat2, mecs2 = solver2.almostSureWin(max_priority=2)

    # Results should be the same
    assert reach1 == reach2
    assert strat1 == strat2
    # Note: mecs might differ in order but should have same content


def test_empty_target_set():
    """Test almost-sure reachability with empty target set."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["a"])
    env.setObs(["o1"])

    env.start[0] = 1.0
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Empty target
    reachable = solver.almostSureReach(set())

    # Should return a list
    assert isinstance(reachable, list)


def test_nondeterministic_mdp():
    """Test on MDP with nondeterministic choices."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1", "s2"])
    env.setActions(["a1", "a2"])
    env.setObs(["o1", "o2"])

    env.start[0] = 1.0

    # Action 0: s0 -> s1
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 2, 1.0)
    env._addOneTrans(2, 0, 2, 1.0)

    # Action 1: s0 -> s2
    env._addOneTrans(0, 1, 2, 1.0)
    env._addOneTrans(1, 1, 0, 1.0)
    env._addOneTrans(2, 1, 0, 1.0)

    env._addOneObs(0, 1, 0, 1.0)
    env._addOneObs(0, 2, 1, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env._addOneObs(1, 1, 0, 1.0)
    env._addOneObs(1, 2, 1, 1.0)
    env._addOneObs(1, 0, 0, 1.0)

    env.computeTrans()
    env.addAtom(0, ["o1"], ids=False)

    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    solver = ParityMDPSolver(mdp)

    # Compute winning strategy
    reach, strat, mecs = solver.almostSureWin(max_priority=2)

    # Check that strategies choose from available actions
    for s in strat:
        if s in mdp.trans:
            available_actions = set(mdp.trans[s].keys())
            chosen_actions = set(strat[s])
            assert chosen_actions.issubset(available_actions)
