# POMDPs with Reveals
This is an implementation of algorithms for POMDPs which have the property of
revealing the state every so often.

## Simulating POMDPs
You can use `sim.py` to simulate a given POMDP. You will have to choose
actions to step through the environment and you will only get as feedback the
(support) of the beliefs. Internally, the simulator will use a pseudo-random
number generator to select a concrete successor state.

## Visualizing POMDPs
You can use `vis.py` to visualize a given POMDP. In addition, you can also
visualize the belief-support automaton on which the analysis of almost-sure
winning is realized. Check the help message of the script to determine what
arguments it requires.
