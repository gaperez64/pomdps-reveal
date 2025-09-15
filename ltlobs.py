#!/usr/bin/python3

import argparse
from io import FileIO, TextIOWrapper

from pomdpy.pomdp import POMDP
from pomdpy.beliefsuppautobs import BeliefSuppAut
from pomdpy.parsers import pomdp
from pomdpy.product import get_state_name
import spot

from typing import Optional


def print_parity(file: TextIOWrapper, aut):
    bdict = aut.get_dict()
    file.write("----  PARITY AUTOMATON  ----\n")
    for s in range(aut.num_states()):
        file.write(f"State {s}\n")
        for t in aut.out(s):
            file.write(f"  edge({t.src} -> {t.dst})\n")
            # bdd_print_formula() is designed to print on a std::ostream, and
            # is inconvenient to use in Python.  Instead we use
            # bdd_format_formula() as this simply returns a string.
            file.write("    label =" +  str(spot.bdd_format_formula(bdict, t.cond)) + "\n")
            file.write("    acc sets =" + str(t.acc) + "\n")
    file.write("\n")

def print_pomdp(file, env):
    file.write("---- POMDP ----\n")
    file.write(f"POMDP states: {env.statesinv}\n")
    file.write(f"POMDP atomic proposotions: {env.atoms}\n")
    file.write(f"POMDP observations: {env.obs}\n{env.obsfun}\n")
    # for obs in env.obsinv.values():
    #     file.write(env.generate_observation_formula(obs) + "\n")
    file.write(f"POMDP actions: {env.actions}\n")
    file.write(f"POMDP transitions: {env.trans}\n")
    file.write("\n")

def print_belief_support(file: TextIOWrapper, aut):
    file.write("---- BELIEF SUPPORT ----\n")
    file.write(f"Belief-support MDP number of states: {len(aut.states)}\n")
    for i, b in enumerate(aut.states):
        file.write(f"Beilef {i}: \n\t")
        for (s, a) in b:
            file.write(f"s{s}-{a} ")
        file.write("\n")
    file.write("Transitions: \n")
    for idx, src in enumerate(aut.trans):
        file.write(f"From {list(aut.states[src])} (B{idx}):\n")
        for act in aut.trans[src]:
            file.write(f"\t Doing {aut.actions[act]}: ")
            for dst in aut.trans[src][act]:
                file.write(f"{list(aut.states[dst])} (B{dst}), ")
            file.write(f"\n")
    file.write(f"Priorities: {aut.prio}")

    file.write("\n")


def asWin(env: POMDP, ltl_formula: str, visualize: Optional[str] = None):

    PRINT_AUT = True
    PRINT_POMDP = True
    PRINT_PRODUCT = True
    PRINT_BELIEF = True
    PRINT_RESULT = True

    with open("debug.txt", "w") as file:
        pass

    # 1. Translate LTL to a parity automaton
    parity_automaton = spot.translate(ltl_formula, "parity", "complete", "SBAcc")
    # Splitting edges makes it easier to find the automaton transition for an observation
    parity_automaton = spot.split_edges(parity_automaton)

    # 2. Construct the belief-support automaton directly.
    # This is the main fix: we no longer build a flawed intermediate product.
    # The constructor now handles the complex interaction between POMDP transitions,
    # observations, and automaton transitions.
    print("Constructing belief-support automaton...")
    aut = BeliefSuppAut(env, parity_automaton)
    print(f"Belief-support automaton constructed with {len(aut.states)} states.")

    with open("debug.txt", "a") as file:
        if PRINT_AUT:
            print_parity(file, parity_automaton)
        if PRINT_POMDP:
            print_pomdp(file, env)
        # if PRINT_PRODUCT:
        #     print_product(product, env, parity_automaton)
        if PRINT_BELIEF:
            print_belief_support(file, aut) 



    # 3. Solve the parity game on the belief-support automaton.
    max_prio = 0
    if aut.prio:
        max_prio = max(aut.prio.values())

    print(f"Solving parity game with max priority {max_prio}...")
    (aswin, reach_strat, mec_strats) = aut.almostSureWin(
        max_priority=max_prio, vis=visualize
    )
    print("Done solving.")

    # 4. Print the results
    # Helper to get human-readable names for product states (pomdp_state, aut_state)
    def get_prod_state_name(s_idx, q_idx):
        return f"{env.states[s_idx]}-{q_idx}"
    if PRINT_RESULT:
        print("\n--- Winning Strategy ---")

        # Print the reachability part of the strategy
        print(f"\nReachability Strategy (for {len(reach_strat)} beliefs):")
        for belief_idx, actions in sorted(reach_strat.items()):
            # aut.states[belief_idx] is a tuple of (s, q) tuples
            state_names = [get_prod_state_name(s, q) for s, q in aut.states[belief_idx]]
            print(f"  From B{belief_idx} {state_names}")
            print(f"    -> Use Actions: {[env.actions[idx] for idx in actions]}")

        # Print the MEC-staying parts of the strategy
        for i, mec_strat in enumerate(mec_strats):
            # The priority of the MEC is determined by the max priority of its states.
            # This is a simplification; goodMECs finds MECs for a specific even priority.
            mec_prio = "N/A"
            if mec_strat:
                first_belief = next(iter(mec_strat))
                mec_prio = aut.prio[first_belief]

            print(f"\nMEC Strategy (Priority {mec_prio}, for {len(mec_strat)} beliefs):")
            for belief_idx, actions in sorted(mec_strat.items()):
                state_names = [get_prod_state_name(s, q) for s, q in aut.states[belief_idx]]
                print(f"  In B{belief_idx} {state_names}")
                print(f"    -> Use Actions: {[env.actions[idx] for idx in actions]}")

        if visualize:
            print(f"\nVisualization of the strategy graph saved to '{visualize}'")


# Set up the parse arguments
parser = argparse.ArgumentParser(
    prog="ltl.py", description="Almost-sure LTL model checking for POMDPs"
)
parser.add_argument("filename", help="POMDP filename")
parser.add_argument(
    "-l",
    "--ltl",
    action="store",
    required=True,
    help="LTL formula with atomic propositions over observations",
)
parser.add_argument(
    "-v",
    "--visualize",
    action="store",
    required=False,
    default=None,
    help="Output winning region graph to a file (e.g., 'strategy.png' or 'strategy.dot')",
)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.filename) as f:
        env = pomdp.parse(f.read())
    asWin(env, args.ltl, args.visualize)
    exit(0)
