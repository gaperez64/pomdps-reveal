#!/usr/bin/python3

import sys

from pomdpy.parsers import pomdp


def simulate(filename):
    with open(filename) as f:
        env = pomdp.parse(f.read())
    env.reset()
    prev = None
    while True:
        print(f"Enter an action: {env.actions}")
        act = input()
        if act == "" and prev is not None:
            act = prev
        elif act == "" or act not in env.actions:
            continue
        obs = env.step(act)
        print(f"New observation = {obs}")
        if obs is None:
            print("A sink was reached, simulation stopped.")
            return
        prev = act


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{sys.argv[0]} expected exactly one argument: POMDP filename")
        exit(1)
    simulate(sys.argv[1])
    exit(0)
