#!/usr/bin/env python3
"""
Regenerate expected regression results for Almost-Sure Winning and MEC tests.

This script rebuilds the ProductPOMDP and Belief-Support MDP with the current
implementation and writes fresh pickles under `pomdpy/tests/expected_results`.

Usage:
  uv run python scripts/regenerate_expected_results.py
"""

import os
import sys
import pickle
import re
import spot

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.product import ProductPOMDP
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver

EXPECTED_DIR = os.path.join(ROOT, "pomdpy", "tests", "expected_results")


def parse_tlsf_file(tlsf_path):
    """Parse TLSF file to extract LTL formulas."""
    with open(tlsf_path, 'r') as f:
        content = f.read()
    
    result = {'formulas': []}
    
    # Extract GUARANTEES section
    guarantees_match = re.search(r'GUARANTEES\s*\{([^}]+)\}',
                                  content, re.DOTALL)
    if guarantees_match:
        guarantees_text = guarantees_match.group(1)
        lines = guarantees_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('//') or not line:
                continue
            if line.endswith(';'):
                formula = line[:-1].strip()
                # Convert TLSF operators and remove escapes
                formula = formula.replace('&&', '&').replace('||', '|')
                formula = formula.replace('\\!', '!')
                formula = formula.replace('\\&', '&')
                formula = formula.replace('\\|', '|')
                if formula:
                    result['formulas'].append(formula)
    
    return result


# Test cases: (POMDP path, TLSF path, result filename)
CASES_ASWIN = [
    ("examples/ltl/ltl-revealing-tiger.pomdp",
     "examples/ltl/ltl-revealing-tiger.tlsf",
     "revealing_tiger_aswin.pkl"),
    ("examples/ltl/ltl-corridor-easy.pomdp",
     "examples/ltl/ltl-corridor-easy.tlsf",
     "corridor_easy_aswin.pkl"),
    ("examples/ltl/ltl-revealing-tiger-repeating.pomdp",
     "examples/ltl/ltl-revealing-tiger-repeating.tlsf",
     "tiger_repeating_aswin.pkl"),
]

CASES_MEC = [
    ("examples/ltl/ltl-revealing-tiger.pomdp",
     "examples/ltl/ltl-revealing-tiger.tlsf",
     "revealing_tiger_mec.pkl"),
    ("examples/ltl/ltl-corridor-easy.pomdp",
     "examples/ltl/ltl-corridor-easy.tlsf",
     "corridor_easy_mec.pkl"),
    ("examples/ltl/ltl-revealing-tiger-repeating.pomdp",
     "examples/ltl/ltl-revealing-tiger-repeating.tlsf",
     "tiger_repeating_mec.pkl"),
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def translate(formula: str):
    aut = spot.translate(formula, "parity", "complete", "SBAcc")
    return spot.split_edges(aut)


def run_aswin(pomdp_path: str, tlsf_path: str):
    # Load POMDP
    with open(os.path.join(ROOT, pomdp_path), "r") as f:
        env = pomdp_parser.parse(f.read())
    
    # Load formula from TLSF
    tlsf_data = parse_tlsf_file(os.path.join(ROOT, tlsf_path))
    if not tlsf_data['formulas']:
        raise ValueError(f"No formulas found in {tlsf_path}")
    
    # Combine all formulas with &
    if len(tlsf_data['formulas']) > 1:
        formula = " & ".join(f"({f})" for f in tlsf_data['formulas'])
    else:
        formula = tlsf_data['formulas'][0]
    
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
        'formula': formula,
    }


def run_mec(pomdp_path: str, tlsf_path: str):
    # Load POMDP
    with open(os.path.join(ROOT, pomdp_path), "r") as f:
        env = pomdp_parser.parse(f.read())
    
    # Load formula from TLSF
    tlsf_data = parse_tlsf_file(os.path.join(ROOT, tlsf_path))
    if not tlsf_data['formulas']:
        raise ValueError(f"No formulas found in {tlsf_path}")
    
    # Combine all formulas with &
    if len(tlsf_data['formulas']) > 1:
        formula = " & ".join(f"({f})" for f in tlsf_data['formulas'])
    else:
        formula = tlsf_data['formulas'][0]
    
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
        'formula': formula,
    }


def main():
    ensure_dir(EXPECTED_DIR)

    print("Regenerating Almost-Sure Winning results...")
    for path, tlsf_path, fname in CASES_ASWIN:
        print(f"\n{fname}:")
        print(f"  POMDP: {path}")
        print(f"  TLSF: {tlsf_path}")
        data = run_aswin(path, tlsf_path)
        with open(os.path.join(EXPECTED_DIR, fname), 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Wrote {fname}")
        print(f"    Winning POMDP states: {data['winning_pomdp_states']}")
        print(f"    BS states: {len(data['winning_bs_states'])}")

    print("\n" + "="*60)
    print("Regenerating MEC results...")
    for path, tlsf_path, fname in CASES_MEC:
        print(f"\n{fname}:")
        print(f"  POMDP: {path}")
        print(f"  TLSF: {tlsf_path}")
        data = run_mec(path, tlsf_path)
        with open(os.path.join(EXPECTED_DIR, fname), 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Wrote {fname}")
        print(f"    Max priority: {data['max_priority']}")
        print(f"    BS states: {data['num_states']}")

    print("\n" + "="*60)
    print("Done! All expected results regenerated.")


if __name__ == "__main__":
    main()
