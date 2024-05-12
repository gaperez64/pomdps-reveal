#!/usr/bin/python3

import argparse

from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.parsers import pomdp


def visBeliefSupp(env, out):
    BeliefSuppAut(env).show(out)


def visPOMDP(env, out):
    env.show(out)


# Set up the parse arguments
parser = argparse.ArgumentParser(
                    prog="vis.py",
                    description="Creates graphviz depiction of (PO)MDPs")
parser.add_argument("filename",            # positional argument
                    help="POMDP filename")
parser.add_argument("output",              # positional argument
                    help="output filename (png or dot)")
parser.add_argument("-s", "--beliefsupp",  # on/off flag
                    action="store_true",
                    help="Show belief-support automaton")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.filename) as f:
        env = pomdp.parse(f.read())
    if args.beliefsupp:
        visBeliefSupp(env, args.output)
    else:
        visPOMDP(env, args.output)
    exit(0)
