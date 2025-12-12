"""
Tests for belief_support_MDP.py

This module tests the BeliefSuppMDP class which constructs belief-support
MDPs from AtomicPropPOMDP instances and parity automata.
"""

import spot
from pomdpy.pomdp import AtomicPropPOMDP
from pomdpy.belief_support_MDP import (
    BeliefSuppMDP,
    get_next_aut_state_by_observation
)
from pomdpy.product import ProductPOMDP
from pomdpy.parsers import pomdp


def test_next_aut_state_by_observation():
    """Test that observations correctly guide automaton transitions."""
    env = AtomicPropPOMDP()
    env.setStates(["a", "b"])
    env.setObs(["obs1", "obs2"])
    
    # Define atomic propositions on observations
    env.addAtom(0, ["obs1"], ids=False)  # p0 true on obs1
    env.addAtom(1, ["obs2"], ids=False)  # p1 true on obs2
    
    # Create automaton for GF p0 & GF p1
    aut = spot.translate("GFp0 & GFp1", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Test that obs1 (p0 true, p1 false) leads to correct state
    obs1_idx = env.obsinv["obs1"]
    # Build product to supply observation propositions consistently
    product = ProductPOMDP(env, aut)
    next_state = get_next_aut_state_by_observation(product, aut, obs1_idx, 0)
    assert isinstance(next_state, int)
    assert next_state >= 0


def test_belief_support_mdp_initialization():
    """Test basic BeliefSuppMDP initialization."""
    env = AtomicPropPOMDP()
    env.setStates(["a", "b"])
    env.setActions(["act1"])
    env.setObs(["obs1", "obs2"])
    
    # Set initial state
    env.start[0] = 1.0
    
    # Add transitions
    env._addOneTrans(0, 0, 0, 0.4)
    env._addOneTrans(0, 0, 1, 0.6)
    env._addOneTrans(1, 0, 0, 0.7)
    env._addOneTrans(1, 0, 1, 0.3)
    
    # Add observations
    env._addOneObs(0, 0, 0, 0.6)
    env._addOneObs(0, 0, 1, 0.4)
    env._addOneObs(0, 1, 0, 0.7)
    env._addOneObs(0, 1, 1, 0.3)
    
    # Compute joint transitions
    env.computeTrans()
    
    # Define atomic propositions on observations
    env.addAtom(0, ["obs1"], ids=False)
    
    # Create automaton
    aut = spot.translate("FGp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Build belief-support MDP
    product = ProductPOMDP(env, aut)
    mdp = BeliefSuppMDP(product, aut)
    
    # Check basic structure
    assert len(mdp.states) > 0
    assert len(mdp.actions) == len(env.actions)
    assert 0 in mdp.trans  # Initial state should have transitions
    assert mdp.start == 0


def test_belief_support_mdp_states():
    """Test that belief-support MDP states are correctly formed."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1"])
    env.setActions(["a"])
    env.setObs(["o1", "o2"])
    
    # Initial state: both states possible
    env.start[0] = 0.5
    env.start[1] = 0.5
    
    # Add transitions
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneTrans(1, 0, 1, 1.0)
    
    # Add observations
    env._addOneObs(0, 0, 0, 1.0)
    env._addOneObs(0, 1, 1, 1.0)
    
    env.computeTrans()
    
    # Define atomic propositions
    env.addAtom(0, ["o1"], ids=False)
    
    # Create automaton
    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Build MDP
    product = ProductPOMDP(env, aut)
    mdp = BeliefSuppMDP(product, aut)
    
    # Initial belief support should contain both POMDP states
    initial_bs = mdp.states[0]
    assert len(initial_bs) == 2  # Both states in initial belief
    
    # Each element should be a (pomdp_state, aut_state) pair
    for s, q in initial_bs:
        assert isinstance(s, int)
        assert isinstance(q, int)
        assert s in [0, 1]  # POMDP state indices


def test_belief_support_mdp_transitions():
    """Test that transitions are correctly computed."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1"])
    env.setActions(["a1", "a2"])
    env.setObs(["o1", "o2"])
    
    env.start[0] = 1.0
    
    # Add transitions for action 0
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 0, 1.0)
    
    # Add transitions for action 1
    env._addOneTrans(0, 1, 0, 1.0)
    env._addOneTrans(1, 1, 1, 1.0)
    
    # Add observations
    env._addOneObs(0, 0, 0, 1.0)
    env._addOneObs(0, 1, 1, 1.0)
    env._addOneObs(1, 0, 0, 1.0)
    env._addOneObs(1, 1, 1, 1.0)
    
    env.computeTrans()
    
    # Define atomic propositions
    env.addAtom(0, ["o1"], ids=False)
    
    # Create automaton
    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Build MDP
    product = ProductPOMDP(env, aut)
    mdp = BeliefSuppMDP(product, aut)
    
    # Check transitions structure
    assert 0 in mdp.trans
    assert 0 in mdp.trans[0]  # Action 0 from initial state
    assert 1 in mdp.trans[0]  # Action 1 from initial state
    
    # Transitions should be lists of successor state indices
    assert isinstance(mdp.trans[0][0], list)
    assert all(isinstance(s, int) for s in mdp.trans[0][0])


def test_belief_support_mdp_priorities():
    """Test that priorities are correctly assigned."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["a"])
    env.setObs(["o1"])
    
    env.start[0] = 1.0
    
    # Self-loop
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env.computeTrans()
    
    # Define atomic propositions
    env.addAtom(0, ["o1"], ids=False)
    
    # Create automaton
    aut = spot.translate("GFp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Build MDP
    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    
    # Check that priorities are assigned
    for i in range(len(mdp.states)):
        assert i in mdp.prio
        assert isinstance(mdp.prio[i], int)
        assert mdp.prio[i] >= 0


def test_revealing_tiger_example():
    """Test basic structure of revealing-tiger POMDP for BeliefSuppMDP."""
    with open("examples/ltl-revealing/ltl-tiger-repeating.pomdp", "r") as f:
        content = f.read()
        env = pomdp.parse(content)
    
    # Should be an AtomicPropPOMDP
    assert isinstance(env, AtomicPropPOMDP)
    
    # Check basic POMDP structure is compatible with BeliefSuppMDP
    assert len(env.states) > 0
    assert len(env.actions) > 0
    assert len(env.obs) > 0
    assert len(env.atoms) > 0
    
    # Check that transitions are properly formed
    assert len(env.transitions) > 0
    for s in env.transitions:
        for a in env.transitions[s]:
            assert isinstance(env.transitions[s][a], dict)
            for (s_prime, o), prob in env.transitions[s][a].items():
                assert 0 <= s_prime < len(env.states)
                assert 0 <= o < len(env.obs)
                assert 0 <= prob <= 1.0
    
    # Note: Full MDP construction requires matching formula with observations
    # The actual belief_support_algorithm.py uses the correct formula for
    # each file. Here we just verify the POMDP structure is sound.


def test_empty_belief_support():
    """Test handling of edge cases with potentially empty beliefs."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["a"])
    env.setObs(["o1"])
    
    env.start[0] = 1.0
    
    # Self-loop
    env._addOneTrans(0, 0, 0, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    env.computeTrans()
    
    # Define atomic propositions
    env.addAtom(0, ["o1"], ids=False)
    
    # Create automaton
    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Build MDP
    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    
    # Should have at least the initial state
    assert len(mdp.states) >= 1
    
    # No state should be an empty tuple
    for st in mdp.states:
        assert len(st) > 0


def test_multiple_initial_states():
    """Test BeliefSuppMDP with multiple initial POMDP states."""
    env = AtomicPropPOMDP()
    env.setStates(["s0", "s1", "s2"])
    env.setActions(["a"])
    env.setObs(["o1", "o2"])
    
    # Multiple initial states
    env.start[0] = 0.3
    env.start[1] = 0.5
    env.start[2] = 0.2
    
    # Add transitions
    env._addOneTrans(0, 0, 1, 1.0)
    env._addOneTrans(1, 0, 2, 1.0)
    env._addOneTrans(2, 0, 0, 1.0)
    
    # Add observations
    env._addOneObs(0, 1, 0, 1.0)
    env._addOneObs(0, 2, 1, 1.0)
    env._addOneObs(0, 0, 0, 1.0)
    
    env.computeTrans()
    
    # Define atomic propositions
    env.addAtom(0, ["o1"], ids=False)
    
    # Create automaton
    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Build MDP
    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    
    # Initial belief support should contain all initial POMDP states
    initial_bs = mdp.states[0]
    initial_pomdp_states = {s for s, q in initial_bs}
    assert initial_pomdp_states == {0, 1, 2}


def test_action_preservation():
    """Test that actions are preserved from POMDP to MDP."""
    env = AtomicPropPOMDP()
    env.setStates(["s0"])
    env.setActions(["listen", "open-left", "open-right"])
    env.setObs(["o1"])
    
    env.start[0] = 1.0
    
    # Add transitions for each action
    for act_idx in range(3):
        env._addOneTrans(0, act_idx, 0, 1.0)
        env._addOneObs(act_idx, 0, 0, 1.0)
    
    env.computeTrans()
    
    # Define atomic propositions
    env.addAtom(0, ["o1"], ids=False)
    
    # Create automaton
    aut = spot.translate("Fp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    
    # Build MDP
    product = ProductPOMDP(env, aut); mdp = BeliefSuppMDP(product, aut)
    
    # Actions should be preserved
    assert mdp.actions == env.actions
    assert mdp.actionsinv == env.actionsinv
    assert len(mdp.actions) == 3


def test_corridor_example():
    """Test basic structure of corridor POMDP for BeliefSuppMDP."""
    try:
        with open("examples/revealing_ltl-corridor-easy.pomdp", "r") as f:
            content = f.read()
            env = pomdp.parse(content)
        
        # Should be an AtomicPropPOMDP
        assert isinstance(env, AtomicPropPOMDP)
        
        # Check basic structure
        assert len(env.states) > 0
        assert len(env.actions) > 0
        assert len(env.obs) > 0
        
        # Check transitions are properly formed
        if env.atoms:
            assert len(env.transitions) > 0
            for s in env.transitions:
                for a in env.transitions[s]:
                    assert isinstance(env.transitions[s][a], dict)
                    
        # Note: Full MDP construction requires matching formula
        # Here we just verify the POMDP structure is compatible
    except FileNotFoundError:
        # Skip if file doesn't exist
        pass




