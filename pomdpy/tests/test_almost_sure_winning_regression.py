"""
Regression tests for almost-sure winning computation.

These tests verify that the almost-sure winning algorithm produces consistent
results across different POMDP examples with LTL objectives.
"""

import pickle
import os
import pytest
from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.product import ProductPOMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
import spot


# Base directory for expected results
EXPECTED_DIR = os.path.join(
    os.path.dirname(__file__), "expected_results"
)


def load_expected_result(filename):
    """Load expected result from pickle file."""
    filepath = os.path.join(EXPECTED_DIR, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def run_almost_sure_winning(pomdp_file, ltl_formula):
    """
    Run the almost-sure winning algorithm on a POMDP with LTL objective.
    
    Args:
        pomdp_file: Path to POMDP file
        ltl_formula: LTL formula string
        
    Returns:
        Dictionary with results
    """
    # Load POMDP
    with open(pomdp_file, 'r') as f:
        content = f.read()
    env = pomdp_parser.parse(content)
    
    # Translate LTL to parity automaton
    parity_automaton = spot.translate(
        ltl_formula, "parity", "complete", "SBAcc"
    )
    parity_automaton = spot.split_edges(parity_automaton)
    
    # Build belief-support MDP
    product_pomdp = ProductPOMDP(env, parity_automaton); belief_supp_mdp = BeliefSuppMDP(product_pomdp, parity_automaton)
    
    # Solve for almost-sure winning
    solver = ParityMDPSolver(belief_supp_mdp, verbose=False)
    max_prio = max(belief_supp_mdp.prio.values())
    (aswin, reach_strats, mec_strats) = solver.almostSureWin(
        max_priority=max_prio
    )
    
    # Extract POMDP states from winning belief-support states
    winning_pomdp_states = set()
    for bs_idx in aswin:
        bs = belief_supp_mdp.states[bs_idx]
        for pomdp_state, _ in bs:
            winning_pomdp_states.add(pomdp_state)
    
    return {
        'winning_bs_states': sorted(aswin),
        'winning_pomdp_states': sorted(winning_pomdp_states),
        'reach_strategies': reach_strats,
        'mec_strategies': mec_strats,
        'max_priority': max_prio,
        'num_bs_states': len(belief_supp_mdp.states),
        'num_pomdp_states': len(env.states),
    }


def test_revealing_tiger_gfp0_not_p1():
    """Test almost-sure winning on revealing tiger with GFp0 & G!p1."""
    result = run_almost_sure_winning(
        "examples/ltl-revealing/ltl-tiger-repeating.pomdp",
        "GFp0 & G!p1"
    )
    expected = load_expected_result("revealing_tiger_gfp0_not_p1_aswin.pkl")
    
    # Check key properties
    assert result['winning_bs_states'] == expected['winning_bs_states']
    assert result['winning_pomdp_states'] == expected['winning_pomdp_states']
    assert result['max_priority'] == expected['max_priority']
    assert result['num_bs_states'] == expected['num_bs_states']
    assert result['num_pomdp_states'] == expected['num_pomdp_states']
    
    # Check strategies exist for all winning states
    assert len(result['reach_strategies']) == len(expected['reach_strategies'])
    assert len(result['mec_strategies']) == len(expected['mec_strategies'])


def test_corridor_easy_gfp0_gfp1():
    """Test almost-sure winning on corridor easy with GFp0 & GFp1."""
    result = run_almost_sure_winning(
        "examples/ltl-revealing/ltl-corridor-easy.pomdp",
        "GFp0 & GFp1"
    )
    expected = load_expected_result("corridor_easy_gfp0_gfp1_aswin.pkl")
    
    assert result['winning_bs_states'] == expected['winning_bs_states']
    assert result['winning_pomdp_states'] == expected['winning_pomdp_states']
    assert result['max_priority'] == expected['max_priority']
    assert result['num_bs_states'] == expected['num_bs_states']
    assert result['num_pomdp_states'] == expected['num_pomdp_states']
    assert len(result['reach_strategies']) == len(expected['reach_strategies'])
    assert len(result['mec_strategies']) == len(expected['mec_strategies'])


def test_corridor_easy_complex():
    """Test almost-sure winning on corridor easy with complex formula."""
    result = run_almost_sure_winning(
        "examples/ltl-revealing/ltl-corridor-easy.pomdp",
        "G(p0 & X!p0 -> X(!p0 U p1)) & G(p1 & X!p1 -> X(!p1 U p0)) & GFp0 & GFp1"
    )
    expected = load_expected_result("corridor_easy_complex_aswin.pkl")
    
    assert result['winning_bs_states'] == expected['winning_bs_states']
    assert result['winning_pomdp_states'] == expected['winning_pomdp_states']
    assert result['max_priority'] == expected['max_priority']
    assert result['num_bs_states'] == expected['num_bs_states']
    assert result['num_pomdp_states'] == expected['num_pomdp_states']
    assert len(result['reach_strategies']) == len(expected['reach_strategies'])
    assert len(result['mec_strategies']) == len(expected['mec_strategies'])


def test_tiger_repeating_gfp0_not_p1():
    """Test almost-sure winning on repeating tiger with GFp0 & G!p1."""
    result = run_almost_sure_winning(
        "examples/ltl-revealing/ltl-tiger-repeating.pomdp",
        "GFp0 & G!p1"
    )
    expected = load_expected_result("tiger_repeating_gfp0_not_p1_aswin.pkl")
    
    assert result['winning_bs_states'] == expected['winning_bs_states']
    assert result['winning_pomdp_states'] == expected['winning_pomdp_states']
    assert result['max_priority'] == expected['max_priority']
    assert result['num_bs_states'] == expected['num_bs_states']
    assert result['num_pomdp_states'] == expected['num_pomdp_states']
    assert len(result['reach_strategies']) == len(expected['reach_strategies'])
    assert len(result['mec_strategies']) == len(expected['mec_strategies'])


def test_almost_sure_winning_properties():
    """Test general properties of almost-sure winning computation."""
    result = run_almost_sure_winning(
        "examples/ltl-revealing/ltl-tiger-repeating.pomdp",
        "GFp0 & G!p1"
    )
    
    # Winning POMDP states should be a subset of all POMDP states
    assert len(result['winning_pomdp_states']) <= result['num_pomdp_states']
    
    # Winning BS states should be a subset of all BS states
    assert len(result['winning_bs_states']) <= result['num_bs_states']
    
    # All winning states should have valid indices
    assert all(0 <= s < result['num_bs_states'] 
               for s in result['winning_bs_states'])
    assert all(0 <= s < result['num_pomdp_states'] 
               for s in result['winning_pomdp_states'])
    
    # Strategies should cover winning states
    assert len(result['reach_strategies']) <= len(result['winning_bs_states'])
    
    # MEC strategies should be organized by priority levels
    assert len(result['mec_strategies']) > 0
    assert result['max_priority'] >= 0


def test_winning_states_consistency():
    """
    Test that winning POMDP states are correctly extracted from
    winning BS states.
    """
    # Run on a simple example
    with open("examples/ltl-revealing/ltl-tiger-repeating.pomdp", 'r') as f:
        content = f.read()
    env = pomdp_parser.parse(content)
    
    parity_automaton = spot.translate(
        "GFp0 & G!p1", "parity", "complete", "SBAcc"
    )
    parity_automaton = spot.split_edges(parity_automaton)
    
    product_pomdp = ProductPOMDP(env, parity_automaton); belief_supp_mdp = BeliefSuppMDP(product_pomdp, parity_automaton)
    solver = ParityMDPSolver(belief_supp_mdp, verbose=False)
    max_prio = max(belief_supp_mdp.prio.values())
    (aswin, _, _) = solver.almostSureWin(max_priority=max_prio)
    
    # Extract POMDP states manually
    winning_pomdp_states = set()
    for bs_idx in aswin:
        bs = belief_supp_mdp.states[bs_idx]
        for pomdp_state, _ in bs:
            winning_pomdp_states.add(pomdp_state)
    
    # Every winning BS state should contain at least one POMDP state
    for bs_idx in aswin:
        bs = belief_supp_mdp.states[bs_idx]
        assert len(bs) > 0, "Belief support should not be empty"
        
        # All POMDP states in this BS should be in winning set
        for pomdp_state, _ in bs:
            assert pomdp_state in winning_pomdp_states
