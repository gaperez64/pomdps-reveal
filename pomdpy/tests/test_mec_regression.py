"""
Regression tests for MEC (Maximal End Component) computation.

These tests verify that the MEC computation algorithm produces consistent
results across different POMDP examples with LTL objectives.
"""

import pickle
import os
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


def run_mec_computation(pomdp_file, ltl_formula):
    """
    Run MEC computation on a POMDP with LTL objective.
    
    Args:
        pomdp_file: Path to POMDP file
        ltl_formula: LTL formula string
        
    Returns:
        Dictionary with MEC results for each priority level
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
    
    # Create solver and compute MECs
    solver = ParityMDPSolver(belief_supp_mdp, verbose=False)
    max_prio = max(belief_supp_mdp.prio.values())
    
    # Compute MECs for each even priority level
    mec_results = {}
    for prio in range(max_prio + 1):
        if prio % 2 == 0:  # Even priorities (good for Player 0)
            mecs, strategy = solver.goodMECs(prio)
            mec_results[prio] = {
                'mecs': [list(mec) for mec in mecs],
                'strategy': {k: list(v) for k, v in strategy.items()}
            }
    
    return {
        'mecs': mec_results,
        'max_priority': max_prio,
        'num_states': len(belief_supp_mdp.states),
    }


def test_mec_revealing_tiger_gfp0_not_p1():
    """Test MEC computation on revealing tiger with GFp0 & G!p1."""
    result = run_mec_computation(
        "examples/ltl/ltl-revealing-tiger.pomdp",
        "GFp0 & G!p1"
    )
    expected = load_expected_result("revealing_tiger_gfp0_not_p1_mec.pkl")
    
    # Check basic properties
    assert result['max_priority'] == expected['max_priority']
    assert result['num_states'] == expected['num_states']
    
    # Check MECs for each priority
    assert set(result['mecs'].keys()) == set(expected['mecs'].keys())
    for prio in result['mecs']:
        result_mec_list = result['mecs'][prio]['mecs']
        expected_mec_list = expected['mecs'][prio]['mecs']
        assert len(result_mec_list) == len(expected_mec_list)
        
        # Check that each MEC has the same states
        result_mecs_set = {frozenset(mec) for mec in result_mec_list}
        expected_mecs_set = {frozenset(mec) for mec in expected_mec_list}
        assert result_mecs_set == expected_mecs_set


def test_mec_corridor_easy_gfp0_gfp1():
    """Test MEC computation on corridor easy with GFp0 & GFp1."""
    result = run_mec_computation(
        "examples/ltl/ltl-corridor-easy.pomdp",
        "GFp0 & GFp1"
    )
    expected = load_expected_result("corridor_easy_gfp0_gfp1_mec.pkl")
    
    assert result['max_priority'] == expected['max_priority']
    assert result['num_states'] == expected['num_states']
    
    assert set(result['mecs'].keys()) == set(expected['mecs'].keys())
    for prio in result['mecs']:
        assert len(result['mecs'][prio]) == len(expected['mecs'][prio])
        result_mecs = [frozenset(mec) for mec in result['mecs'][prio]]
        expected_mecs = [frozenset(mec) for mec in expected['mecs'][prio]]
        assert set(result_mecs) == set(expected_mecs)


def test_mec_corridor_easy_complex():
    """Test MEC computation on corridor easy with complex formula."""
    result = run_mec_computation(
        "examples/ltl/ltl-corridor-easy.pomdp",
        "G(p0 & X!p0 -> X(!p0 U p1)) & G(p1 & X!p1 -> X(!p1 U p0)) & GFp0 & GFp1"
    )
    expected = load_expected_result("corridor_easy_complex_mec.pkl")
    
    assert result['max_priority'] == expected['max_priority']
    assert result['num_states'] == expected['num_states']
    
    assert set(result['mecs'].keys()) == set(expected['mecs'].keys())
    for prio in result['mecs']:
        assert len(result['mecs'][prio]) == len(expected['mecs'][prio])
        result_mecs = [frozenset(mec) for mec in result['mecs'][prio]]
        expected_mecs = [frozenset(mec) for mec in expected['mecs'][prio]]
        assert set(result_mecs) == set(expected_mecs)


def test_mec_tiger_repeating_gfp0_not_p1():
    """Test MEC computation on repeating tiger with GFp0 & G!p1."""
    result = run_mec_computation(
        "examples/ltl/ltl-revealing-tiger-repeating.pomdp",
        "GFp0 & G!p1"
    )
    expected = load_expected_result("tiger_repeating_gfp0_not_p1_mec.pkl")
    
    assert result['max_priority'] == expected['max_priority']
    assert result['num_states'] == expected['num_states']
    
    assert set(result['mecs'].keys()) == set(expected['mecs'].keys())
    for prio in result['mecs']:
        assert len(result['mecs'][prio]) == len(expected['mecs'][prio])
        result_mecs = [frozenset(mec) for mec in result['mecs'][prio]]
        expected_mecs = [frozenset(mec) for mec in expected['mecs'][prio]]
        assert set(result_mecs) == set(expected_mecs)


def test_mec_properties():
    """Test general properties of MEC computation."""
    result = run_mec_computation(
        "examples/ltl/ltl-revealing-tiger.pomdp",
        "GFp0 & G!p1"
    )
    
    # MECs should only be computed for even priorities
    for prio in result['mecs']:
        assert prio % 2 == 0, "MECs should only be for even priorities"
    
    # Each MEC should be non-empty
    for prio in result['mecs']:
        for mec in result['mecs'][prio]['mecs']:
            assert len(mec) > 0, "MECs should not be empty"
    
    # All states in MECs should be valid
    for prio in result['mecs']:
        for mec in result['mecs'][prio]['mecs']:
            for state in mec:
                assert 0 <= state < result['num_states']


def test_mec_disjointness():
    """Test that MECs at the same priority level are disjoint."""
    result = run_mec_computation(
        "examples/ltl/ltl-corridor-easy.pomdp",
        "GFp0 & GFp1"
    )
    
    # For each priority level, MECs should be disjoint
    for prio in result['mecs']:
        mecs = result['mecs'][prio]['mecs']
        if len(mecs) > 1:
            for i, mec1 in enumerate(mecs):
                for j, mec2 in enumerate(mecs):
                    if i != j:
                        # MECs should not share states
                        assert len(set(mec1) & set(mec2)) == 0


def test_mec_states_have_correct_priority():
    """Test that states in MECs have the correct priority."""
    # Load POMDP and build MDP
    with open("examples/ltl/ltl-revealing-tiger.pomdp", 'r') as f:
        content = f.read()
    env = pomdp_parser.parse(content)
    
    parity_automaton = spot.translate(
        "GFp0 & G!p1", "parity", "complete", "SBAcc"
    )
    parity_automaton = spot.split_edges(parity_automaton)
    
    product_pomdp = ProductPOMDP(env, parity_automaton)
    belief_supp_mdp = BeliefSuppMDP(product_pomdp, parity_automaton)
    solver = ParityMDPSolver(belief_supp_mdp, verbose=False)
    max_prio = max(belief_supp_mdp.prio.values())
    
    # Compute MECs
    for prio in range(max_prio + 1):
        if prio % 2 == 0:
            mecs, _ = solver.goodMECs(prio)
            
            # All states in these MECs should have priority <= prio
            for mec in mecs:
                if len(mec) == 0:
                    continue
                for state in mec:
                    state_prio = belief_supp_mdp.prio[state]
                    assert state_prio <= prio
                
                # At least one state should have priority == prio
                # (otherwise it would be a MEC for a lower priority)
                max_state_prio = max(
                    belief_supp_mdp.prio[state] for state in mec
                )
                assert max_state_prio == prio


def test_mec_strongly_connected():
    """
    Test that MECs are strongly connected components.
    This is a basic structural test.
    """
    # Load POMDP and build MDP
    with open("examples/ltl/ltl-revealing-tiger.pomdp", 'r') as f:
        content = f.read()
    env = pomdp_parser.parse(content)
    
    parity_automaton = spot.translate(
        "GFp0 & G!p1", "parity", "complete", "SBAcc"
    )
    parity_automaton = spot.split_edges(parity_automaton)
    
    product_pomdp = ProductPOMDP(env, parity_automaton)
    belief_supp_mdp = BeliefSuppMDP(product_pomdp, parity_automaton)
    solver = ParityMDPSolver(belief_supp_mdp, verbose=False)
    max_prio = max(belief_supp_mdp.prio.values())
    
    # Compute MECs
    for prio in range(max_prio + 1):
        if prio % 2 == 0:
            mecs, _ = solver.goodMECs(prio)
            
            for mec in mecs:
                if len(mec) == 0:
                    continue
                mec_set = set(mec)
                
                # From each state in the MEC, there should be at least
                # one action that stays within the MEC
                for state in mec:
                    if state in belief_supp_mdp.trans:
                        has_internal_action = False
                        for action in belief_supp_mdp.trans[state]:
                            successors = belief_supp_mdp.trans[state][action]
                            # Check if all successors are in the MEC
                            if all(s in mec_set for s in successors):
                                has_internal_action = True
                                break
                        
                        # MECs should have internal actions
                        assert has_internal_action or len(mec) == 1
