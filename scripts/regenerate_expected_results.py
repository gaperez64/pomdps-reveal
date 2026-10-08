#!/usr/bin/env python3
"""
Regenerate expected regression results for Almost-Sure Winning and MEC tests.

This script rebuilds the ProductPOMDP and Belief-Support MDP with the current
implementation and writes fresh pickles under `pomdpy/tests/expected_results`.

Usage:
  python scripts/transform_to_revealing.py
  python scripts/regenerate_expected_results.py
"""

import os
import sys
import pickle
import spot

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.product import ProductPOMDP
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver

EXPECTED_DIR = os.path.join(ROOT, "pomdpy", "tests", "expected_results")


# Test cases: (POMDP path, LTL formula, result filename).
# Use the formulas from the regression tests, which differ from the TLSF files.
CASES_ASWIN = [
    ("examples/ltl-revealing/ltl-tiger-repeating.pomdp",
     "GFp0 & G!p1",
     "revealing_tiger_gfp0_not_p1_aswin.pkl"),
    ("examples/ltl-revealing/ltl-corridor-easy.pomdp",
     "GFp0 & GFp1",
     "corridor_easy_gfp0_gfp1_aswin.pkl"),
    ("examples/ltl-revealing/ltl-corridor-easy.pomdp",
     "G(p0 & X!p0 -> X(!p0 U p1)) & G(p1 & X!p1 -> X(!p1 U p0)) & GFp0 & GFp1",
     "corridor_easy_complex_aswin.pkl"),
    ("examples/ltl-revealing/ltl-tiger-repeating.pomdp",
     "GFp0 & G!p1",
     "tiger_repeating_gfp0_not_p1_aswin.pkl"),
]

CASES_MEC = [
    ("examples/ltl-revealing/ltl-tiger-repeating.pomdp",
     "GFp0 & G!p1",
     "tiger_repeating_gfp0_not_p1_mec.pkl"),
    ("examples/ltl-revealing/ltl-corridor-easy.pomdp",
     "GFp0 & GFp1",
     "corridor_easy_gfp0_gfp1_mec.pkl"),
    ("examples/ltl-revealing/ltl-corridor-easy.pomdp",
     "G(p0 & X!p0 -> X(!p0 U p1)) & G(p1 & X!p1 -> X(!p1 U p0)) & GFp0 & GFp1",
     "corridor_easy_complex_mec.pkl"),
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def translate(formula: str):
    aut = spot.translate(formula, "parity", "complete", "SBAcc")
    return spot.split_edges(aut)


def run_aswin(pomdp_path: str, formula: str):
    # Load POMDP
    with open(os.path.join(ROOT, pomdp_path), "r") as f:
        env = pomdp_parser.parse(f.read())

    print(f"  Formula: {formula}")

    aut = translate(formula)
    prod = ProductPOMDP(env, aut)
    bsmdp = BeliefSuppMDP(prod, aut)
    solver = ParityMDPSolver(bsmdp, verbose=False)
    max_prio = max(bsmdp.prio.values())
    aswin, reach_strats, mec_strats = solver.almostSureWin(
        max_priority=max_prio
    )

    # Extract POMDP states
    winning_pomdp_states = set()
    for bs_idx in aswin:
        for s_base, _ in bsmdp.states[bs_idx]:
            winning_pomdp_states.add(s_base)

    return {
        'winning_bs_states': sorted(aswin),
        'winning_pomdp_states': sorted(winning_pomdp_states),
        'reach_strategies': reach_strats,
        'mec_strategies': mec_strats,
        'max_priority': max_prio,
        'num_bs_states': len(bsmdp.states),
        'num_pomdp_states': len(env.states),
    }


def run_mec(pomdp_path: str, formula: str):
    # Load POMDP
    with open(os.path.join(ROOT, pomdp_path), "r") as f:
        env = pomdp_parser.parse(f.read())

    print(f"  Formula: {formula}")

    aut = translate(formula)
    prod = ProductPOMDP(env, aut)
    bsmdp = BeliefSuppMDP(prod, aut)
    solver = ParityMDPSolver(bsmdp, verbose=False)
    max_prio = max(bsmdp.prio.values())

    mec_results = {}
    for prio in range(max_prio + 1):
        if prio % 2 == 0:
            mecs, strategy = solver.goodMECs(prio)
            mec_results[prio] = {
                'mecs': [list(mec) for mec in mecs],
                'strategy': {k: list(v) for k, v in strategy.items()}
            }

    return {
        'mecs': mec_results,
        'max_priority': max_prio,
        'num_states': len(bsmdp.states),
    }


def main():
    ensure_dir(EXPECTED_DIR)

    print("Regenerating Almost-Sure Winning results...")
    for path, formula, fname in CASES_ASWIN:
        print(f"\n{fname}:")
        print(f"  POMDP: {path}")
        data = run_aswin(path, formula)
        with open(os.path.join(EXPECTED_DIR, fname), 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Wrote {fname}")
        print(f"    Winning POMDP states: {data['winning_pomdp_states']}")
        print(f"    BS states: {len(data['winning_bs_states'])}")

    print("\n" + "="*60)
    print("Regenerating MEC results...")
    for path, formula, fname in CASES_MEC:
        print(f"\n{fname}:")
        print(f"  POMDP: {path}")
        data = run_mec(path, formula)
        with open(os.path.join(EXPECTED_DIR, fname), 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Wrote {fname}")
        print(f"    Max priority: {data['max_priority']}")
        print(f"    BS states: {data['num_states']}")

    print("\n" + "="*60)
    print("Done! All expected results regenerated.")


if __name__ == "__main__":
    main()
