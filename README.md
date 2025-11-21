# POMDPs with Reveals

This is a Python-3 implementation of algorithms for POMDPs which have the property of revealing the state every so often. To use the programs described below, we strongly suggest you [use a virtual environment and install the required
dependencies using the provided requirements
file](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

The LTL to automata part is powered by [spot](https://spot.lre.epita.fr/).

## Belief-Support Algorithm

The main algorithm is implemented in `belief_support_algorithm.py`. This computes almost-sure winning policies for POMDPs with LTL objectives using an observation-based approach.

### Usage

**Basic usage:**
```bash
python belief_support_algorithm.py --file <pomdp-file> --ltl_formula "<formula>" [options]
```

**Using TLSF files (recommended):**
```bash
python belief_support_algorithm.py --file <pomdp-file> --tlsf_file <tlsf-file> [options]
```

The TLSF (Temporal Logic Synthesis Format) option automatically extracts the LTL formula and determines the correct atomic propositions from the specification file. See the `examples/ltl/` directory for TLSF files paired with POMDPs.

**Options:**
- `--verbose`: Detailed step-by-step output showing each phase of the algorithm
- `--plot`: Generate visualizations (DOT and PNG files if graphviz is available)
- `--atoms`: Manually specify atomic proposition IDs (usually auto-detected)

**Examples:**
```bash
# Using TLSF file (auto-detects formula and atoms)
python belief_support_algorithm.py --file examples/ltl/ltl-revealing-tiger.pomdp \
    --tlsf_file examples/ltl/ltl-revealing-tiger.tlsf --verbose --plot

# Using explicit formula
python belief_support_algorithm.py --file examples/ltl/ltl-revealing-tiger.pomdp \
    --ltl_formula "Fp0" --atoms 0,1 --verbose
```

### Algorithm Overview

1. **LTL to Parity Automaton**: The LTL formula is translated into a parity automaton using spot. The automaton is completed and edges are split to facilitate matching with observations.

2. **Product POMDP Construction**: A product POMDP is constructed by combining the original POMDP with the parity automaton. States in the product are pairs (POMDP state, automaton state), and transitions synchronize the POMDP actions with automaton transitions based on the atomic propositions satisfied by each POMDP state.

3. **Belief-Support MDP Construction**: A belief-support MDP is constructed from the product POMDP. States in this MDP are belief supports, which are sets of product POMDP states (pairs of POMDP state and automaton state). The construction uses a forward exploration starting from the initial belief, where transitions are determined by:
   - Taking an action in the POMDP
   - Observing the resulting observation
   - Updating both the belief over POMDP states and the automaton state based on the observation's atomic propositions

4. **Parity MDP Solving**: The belief-support MDP is solved as a parity MDP to compute almost-sure winning states and strategies. This uses:
   - Maximal End Component (MEC) decomposition to identify strongly connected components
   - Filtering MECs by their maximum priority
   - Computing almost-sure reachability to good MECs
   - Extracting strategies for reaching and staying within winning regions

### Output

The algorithm reports:
- Almost-sure winning POMDP states (states from which the objective can be achieved with probability 1)
- Number of belief-support states explored
- When using `--plot`, generates:
  - `automaton.dot/png`: The parity automaton from the LTL formula
  - `product_pomdp.dot/png`: The POMDP × automaton product
  - `belief_support_mdp.dot/png`: The belief-support MDP with winning strategies highlighted in green

## Simulating POMDPs

You can use `pomdpy/sim.py` to simulate a given POMDP. You will have to choose
actions to step through the environment and you will only get as feedback the
(support) of the beliefs. Internally, the simulator will use a pseudo-random
number generator to select a concrete successor state.
