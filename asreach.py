#!/usr/bin/python3

import argparse

from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.parsers import pomdp


def asReach(env, states):
    print(f"states = {states}")


# Set up the parse arguments
parser = argparse.ArgumentParser(
                    prog="asreach.py",
                    description="Almost sure reachability analysis")
parser.add_argument("filename",            # positional argument
                    help="POMDP filename")
parser.add_argument("-s", "--beliefsupp",  # on/off flag
                    action="store_true",
                    help="Show belief-support automaton")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.filename) as f:
        env = pomdp.parse(f.read())
    asReach(env, args.states)
    exit(0)
