#!/usr/bin/python3

import argparse

from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.parsers import pomdp


def visBeliefSupp(env):
    BeliefSuppAut(env).show()


def visPOMDP(env):
    env.show()


# Set up the parse arguments
parser = argparse.ArgumentParser(
                    prog="vis.py",
                    description="Shows visual depiction of (PO)MDPs")
parser.add_argument("filename",            # positional argument
                    help="POMDP filename")
parser.add_argument("-s", "--beliefsupp",  # on/off flag
                    action="store_true",
                    help="Show belief-support automaton")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.filename) as f:
        env = pomdp.parse(f.read())
    if args.beliefsupp:
        visBeliefSupp(env)
    else:
        visPOMDP(env)
    exit(0)
