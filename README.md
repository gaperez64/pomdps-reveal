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

The TLSF (Temporal Logic Synthesis Format) option automatically extracts the LTL formula and determines the correct atomic propositions from the specification file. See the `examples/` directory for TLSF files paired with POMDPs.

**Options:**
- `--verbose`: Detailed step-by-step output showing each phase of the algorithm
- `--plot`: Generate visualizations (DOT and PNG files if graphviz is available)
- `--atoms`: Manually specify atomic proposition IDs (usually auto-detected)

**Examples:**
```bash
# Using TLSF file (auto-detects formula and atoms)
python belief_support_algorithm.py --file examples/revealing_ltl-tiger.pomdp \
    --tlsf_file examples/revealing_ltl-tiger.tlsf --verbose --plot

# Using explicit formula
python belief_support_algorithm.py --file examples/revealing_ltl-tiger.pomdp \
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
- Whether the input POMDP is strongly revealing
- When using `--plot`, generates:
  - `automaton.dot/png`: The parity automaton from the LTL formula
  - `product_pomdp.dot/png`: The POMDP × automaton product
  - `belief_support_mdp.dot/png`: The belief-support MDP with winning strategies highlighted in green

### Checking if a POMDP is Revealing

The `pomdpy.revealing` module provides functions to check and transform POMDPs:

```python
from pomdpy.revealing import is_strongly_revealing, make_strongly_revealing
from pomdpy.parsers import pomdp as pomdp_parser

# Load and check a POMDP
with open('examples/ltl-tiger.pomdp', 'r') as f:
    pomdp = pomdp_parser.parse(f.read())

if is_strongly_revealing(pomdp):
    print("POMDP is already strongly revealing")
else:
    print("POMDP is not strongly revealing")
    # Transform it
    revealing_pomdp = make_strongly_revealing(pomdp)
```

### Transforming POMDPs to Strongly Revealing

The transformation algorithm adds new revealing observations for each state, ensuring that any transition that could violate the revealing property gets a unique observation that identifies the target state.

**Using the revealing module directly:**

```bash
# Check if a POMDP is revealing
python -m pomdpy.revealing --file examples/ltl-tiger.pomdp

# Transform a POMDP to be strongly revealing
python -m pomdpy.revealing --file examples/ltl-tiger.pomdp --transform

# Transform and save to a specific file
python -m pomdpy.revealing --file examples/ltl-tiger.pomdp --transform --output examples/revealing_tiger.pomdp

# Transform without checking first (faster for large files)
python -m pomdpy.revealing --file examples/ltl-tiger.pomdp --transform --no-check
```

The module automatically:
- Detects the POMDP type (regular or AtomicPropPOMDP)
- Preserves atomic proposition information for LTL specifications
- Verifies the transformation succeeded
- Skips transformation if the POMDP is already strongly revealing

**Batch transformation script:**

```bash
python scripts/transform_to_revealing.py
```

This script:
- Processes all POMDP files in the `examples/` directory
- Checks if each POMDP is already strongly revealing
- Transforms non-revealing POMDPs by adding revealing observations
- Creates new files with `revealing_` prefix
- Copies matching TLSF files alongside transformed POMDPs
- Skips files larger than 1500 lines or that timeout after 30 seconds
- Preserves atomic proposition information for POMDPs with LTL specifications

### Benchmarking

Run comprehensive benchmarks on all revealing instances:

```bash
python scripts/run_benchmark_revealing.py
```

This script:
- Tests all `revealing_ltl-*.pomdp` files in the `examples/` directory
- Records detailed metrics for each instance:
  - POMDP sizes (states, actions, observations)
  - Automaton sizes (states, edges)
  - Belief-support MDP size
  - Phase-by-phase timing (automaton construction, belief-support construction, solving)
  - Whether the instance is strongly revealing
- Outputs results to `examples/benchmark_revealing.csv`
- Uses a 5-minute timeout per instance

## Simulating POMDPs

You can use `pomdpy/sim.py` to simulate a given POMDP. You will have to choose
actions to step through the environment and you will only get as feedback the
(support) of the beliefs. Internally, the simulator will use a pseudo-random
number generator to select a concrete successor state.
