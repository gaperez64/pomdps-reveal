#!/usr/bin/python3

import sys

from pomdpy.parsers import pomdp


def simulate(filename):
    print(f"opening {filename}")
    with open(filename) as f:
        print(pomdp.parser.parse(f.read()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{sys.argv[0]} expected exactly one argument: POMDP filename")
        exit(1)
    simulate(sys.argv[1])
    exit(0)
