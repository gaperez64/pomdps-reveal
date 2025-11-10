# POMDPs with Reveals

This is a Python-3 implementation of algorithms for POMDPs which have the property of revealing the state every so often. To use the programs described below, we strongly suggest you [use a virtual environment and install the required
dependencies using the provided requirements
file](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

The LTL to automata part is powered by [spot](https://spot.lre.epita.fr/).

## Belief-Support Algorithm

The main algorithm is implemented in `belief_support_algorithm.py`. This computes almost-sure winning policies for POMDPs with LTL objectives using an observation-based approach. The algorithm consists of three main steps:

1. **LTL to Parity Automaton**: The LTL formula is translated into a parity automaton using spot. The automaton is completed and edges are split to facilitate matching with observations.

2. **Belief-Support MDP Construction**: A belief-support MDP is constructed from the POMDP and parity automaton. States in this MDP are belief supports, which are sets of (POMDP state, automaton state) pairs. The construction uses a forward exploration starting from the initial belief, where transitions are determined by:
   - Taking an action in the POMDP
   - Observing the resulting observation
   - Updating both the belief over POMDP states and the automaton state based on the observation's atomic propositions

3. **Parity MDP Solving**: The belief-support MDP is solved as a parity MDP to compute almost-sure winning states and strategies. This uses:
   - Maximal End Component (MEC) decomposition to identify strongly connected components
   - Filtering MECs by their maximum priority
   - Computing almost-sure reachability to good MECs
   - Extracting strategies for reaching and staying within winning regions

The algorithm supports both verbose output for debugging and visualization of the POMDP, automaton, and belief-support MDP with winning strategies highlighted.

### Usage

```bash
python belief_support_algorithm.py <pomdp-file> <ltl-formula> [options]
```

Use `--verbose` for detailed step-by-step output and `--plot` to generate visualizations (DOT and PNG files). This produces DOT and PNG files showing:

- The POMDP structure
- The parity automaton from the LTL formula
- The belief-support MDP with winning strategies highlighted

## Simulating POMDPs

You can use `sim.py` to simulate a given POMDP. You will have to choose
actions to step through the environment and you will only get as feedback the
(support) of the beliefs. Internally, the simulator will use a pseudo-random
number generator to select a concrete successor state.
