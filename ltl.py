#!/usr/bin/python3

import argparse

from pomdpy.pomdp import POMDP
from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.parsers import pomdp
from pomdpy.product import (
    init_product,
    set_start_probs,
    set_transition_probs,
    set_priorities,
    set_obs_probs,
    get_state_name
)
import spot


def asWin(env: POMDP, ltl_formula: str):
    parity_automaton = spot.translate(ltl_formula, "parity", "complete", "SBAcc")
    parity_automaton = spot.split_edges(parity_automaton)
    env.prio = dict(sorted(env.prio.items()))
    print(f"POMDP number of states: {len(env.states)}")
    print(f"Parity automaton number of  states: {parity_automaton.num_states()}")

    product = init_product(env, parity_automaton)
    set_start_probs(product, env, parity_automaton)
    set_transition_probs(product, env, parity_automaton)
    set_priorities(product, env, parity_automaton)
    set_obs_probs(product, env, parity_automaton)

    aut = BeliefSuppAut(product)
    print(f"Belief-support POMDP number of states: {len(product.states)}")

    aut.setPriorities()
    (aswin, *_) = aut.almostSureWin(max_priority=max(aut.prio.keys()))

    # Get the list of states corresponding to each almost-sure winning belief supports
    aswin_states = [aut.states[bs] for bs in aswin]
    # For each set of state indices, get their readable names
    aswin_state_names = [
        [get_state_name(product, env, parity_automaton, idx) for idx in state_set]
        for state_set in aswin_states
    ]
    print(f"Beliefs that can a.s.-win = {aswin_state_names}")

# Set up the parse arguments
parser = argparse.ArgumentParser(
    prog="ltl.py", description="Almost sure LTL analysis"
)
parser.add_argument("filename", help="POMDP filename")  # positional argument
parser.add_argument(
    "-l", "--ltl",  # LTL formula
    action="store",
    required=True,
    help="LTL formula to analyze"
)
parser.add_argument(
    "-v",
    "--visualize",  # optional output filename
    action="store",
    required=False,
    default=None,
    help="output winning region here (png or dot)",
)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.filename) as f:
        env = pomdp.parse(f.read())
    asWin(env, args.ltl)
    exit(0)
