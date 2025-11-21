"""
Generate expected results for almost-sure winning and MEC computation tests.

This script runs the belief-support algorithm on several POMDP examples and
pickles the results for use in regression tests.
"""

import pickle
import os
from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.product import ProductPOMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
import spot


def generate_expected_results():
    """Generate and save expected results for test cases."""
    
    # Test cases: (file, ltl_formula, description)
    test_cases = [
        (
            "examples/ltl-revealing-tiger.pomdp",
            "GFp0 & G!p1",
            "revealing_tiger_gfp0_not_p1"
        ),
        (
            "examples/ltl-corridor-easy.pomdp",
            "GFp0 & GFp1",
            "corridor_easy_gfp0_gfp1"
        ),
        (
            "examples/ltl-corridor-easy.pomdp",
            "G(p0 & X!p0 -> X(!p0 U p1)) & G(p1 & X!p1 -> X(!p1 U p0)) & GFp0 & GFp1",
            "corridor_easy_complex"
        ),
        (
            "examples/ltl-revealing-tiger-repeating.pomdp",
            "GFp0 & G!p1",
            "tiger_repeating_gfp0_not_p1"
        ),
    ]
    
    # Create directory for expected results
    expected_dir = "pomdpy/tests/expected_results"
    os.makedirs(expected_dir, exist_ok=True)
    
    for pomdp_file, ltl_formula, description in test_cases:
        print(f"\nProcessing: {description}")
        print(f"  File: {pomdp_file}")
        print(f"  Formula: {ltl_formula}")
        
        try:
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
            
            # Compute MECs for each priority level
            mec_results = {}
            for prio in range(max_prio + 1):
                if prio % 2 == 0:  # Even priorities
                    mecs, strategy = solver.goodMECs(prio)
                    # Convert sets to lists for serialization
                    mec_results[prio] = {
                        'mecs': [list(mec) for mec in mecs],
                        'strategy': {k: list(v) for k, v in strategy.items()}
                    }
            
            # Save almost-sure winning results
            aswin_result = {
                'winning_bs_states': sorted(aswin),
                'winning_pomdp_states': sorted(winning_pomdp_states),
                'reach_strategies': reach_strats,
                'mec_strategies': mec_strats,
                'max_priority': max_prio,
                'num_bs_states': len(belief_supp_mdp.states),
                'num_pomdp_states': len(env.states),
            }
            
            aswin_filename = f"{expected_dir}/{description}_aswin.pkl"
            with open(aswin_filename, 'wb') as f:
                pickle.dump(aswin_result, f)
            print(f"  Saved almost-sure winning results to: {aswin_filename}")
            print(f"    Winning BS states: {len(aswin)}")
            print(f"    Winning POMDP states: {sorted(winning_pomdp_states)}")
            
            # Save MEC results
            mec_result = {
                'mecs': mec_results,
                'max_priority': max_prio,
                'num_states': len(belief_supp_mdp.states),
            }
            
            mec_filename = f"{expected_dir}/{description}_mec.pkl"
            with open(mec_filename, 'wb') as f:
                pickle.dump(mec_result, f)
            print(f"  Saved MEC results to: {mec_filename}")
            for prio, mecs in mec_results.items():
                print(f"    Priority {prio}: {len(mecs)} MECs")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    generate_expected_results()
    print("\n" + "="*70)
    print("Expected results generation complete!")
    print("="*70)
