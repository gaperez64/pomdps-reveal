#!/usr/bin/python3

import argparse
from itertools import chain, combinations

from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.parsers import pomdp


def powerset(iterable):
    s = list(iterable)
    it = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return list(it)


def asReach(env, targets):
    tids = []
    for t in targets:
        if t not in env.statesinv:
            print(f"Could not find state {t}")
            exit(1)
        tids.append(env.statesinv[t])
    powtids = powerset(tids)

    aut = BeliefSuppAut(env)
    bids = [aut.statesinv[b] for b in powtids
            if b in aut.statesinv]
    asreach = aut.almostSureReach(bids)
    asbfs = [aut.prettyName(aut.states[s]) for s in asreach]
    print(f"Beliefs that can a.s.-reach = {asbfs}")


# Set up the parse arguments
parser = argparse.ArgumentParser(
                    prog="asreach.py",
                    description="Almost sure reachability analysis")
parser.add_argument("filename",            # positional argument
                    help="POMDP filename")
parser.add_argument('-t', '--targets',     # list of targets
                    nargs='+',
                    help='List of target states',
                    required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.filename) as f:
        env = pomdp.parse(f.read())
    asReach(env, args.targets)
    exit(0)
