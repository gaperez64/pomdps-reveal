#!/usr/bin/python3

import argparse

from pomdpy.parsers import pomdp
from pomdpy.env import Env


def simulate(filename, buchi, cobuchi, beliefsupp=False):
    with open(filename) as f:
        env = pomdp.parse(f.read())
    if beliefsupp:
        env = Env(env, buchi, cobuchi)
    env.reset()
    prev = None
    while True:
        print(f"Enter an action: {env.actions}")
        act = input()
        if act == "" and prev is not None:
            act = prev
        elif act == "" or act not in env.actions:
            continue
        prev = act
        act = env.actionsinv[act]
        if beliefsupp:
            (obs, rew, _, _, info) = env.step(act)
            print(f"New observation = {obs}, and reward = {rew}")
            print(f"Add. information = {info}")
        else:
            obs = env.step(act)
            print(f"New observation = {obs}")


# Set up the parse arguments
parser = argparse.ArgumentParser(
                    prog="sim.py",
                    description="Simulates the (PO)MDP")
parser.add_argument("filename",            # positional argument
                    help="POMDP filename")
parser.add_argument("-s", "--beliefsupp",  # on/off flag
                    action="store_true",
                    help="Simulate belief-support automaton")
parser.add_argument('-1', '--cobuchi',     # list of targets
                    nargs='+',
                    help='List of priority-1 states',
                    required=False,
                    default=[])
parser.add_argument('-2', '--buchi',       # list of targets
                    nargs='+',
                    help='List of priority-2 states',
                    required=False,
                    default=[])
if __name__ == "__main__":
    args = parser.parse_args()
    simulate(args.filename, args.buchi, args.cobuchi, args.beliefsupp)
    exit(0)
