#!/usr/bin/python3

import argparse

from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.parsers import pomdp


def asWin(env, buchi, cobuchi):
    aut = BeliefSuppAut(env)
    aut.setBuchi(buchi, cobuchi)
    aswin = aut.almostSureWin()
    # TODO: Make  asWin spit out names directly?
    asbfs = [aut.prettyName(aut.states[s]) for s in aswin]
    print(f"Beliefs that can a.s.-win = {asbfs}")


# Set up the parse arguments
parser = argparse.ArgumentParser(
                    prog="aswin.py",
                    description="Almost sure {0,1,2}-parity analysis")
parser.add_argument("filename",            # positional argument
                    help="POMDP filename")
parser.add_argument('-1', '--cobuchi',     # list of targets
                    nargs='+',
                    help='List of priority-1 states',
                    required=True)
parser.add_argument('-2', '--buchi',       # list of targets
                    nargs='+',
                    help='List of priority-2 states',
                    required=False,
                    default=[])

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.filename) as f:
        env = pomdp.parse(f.read())
    asWin(env, args.buchi, args.cobuchi)
    exit(0)
