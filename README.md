# POMDPs with Reveals

This is a Python-3 implementation of algorithms for POMDPs which have the property of revealing the state every so often. To use the programs described below, we strongly suggest you [use a virtual environment and install the required
dependencies using the provided requirements
file](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

The LTL to automata part is powered by [spot](https://spot.lre.epita.fr/).

## Regenerating Benchmarks

With the dependencies installed, run these commands from the repository root:

```bash
python scripts/convert_pps_to_pomdp.py --dir examples/pps --output examples/parity
python scripts/transform_to_revealing.py
```

The first command converts the PPS inputs into `examples/parity/`. The second
generates revealing models in `examples/parity-revealing/` and
`examples/ltl-revealing/`. Existing output files are overwritten; the original
PPS inputs are preserved. To rerun the experiments, see [Benchmarking](#benchmarking).

## Running Tests

With Python 3.12, the dependencies, and Spot's Python bindings installed, run
from the repository root:

```bash
python scripts/transform_to_revealing.py
python -m pytest
```

The first command generates the models required by the regression tests; skip it
if they have already been generated.

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
python belief_support_algorithm.py --file examples/ltl-revealing/ltl-guard.pomdp \
    --tlsf_file examples/ltl-revealing/ltl-guard.tlsf --verbose --plot

# Using explicit formula
python belief_support_algorithm.py --file examples/ltl-revealing/ltl-guard.pomdp \
    --ltl_formula "Fp0" --atoms 0,1 --verbose
```

### Algorithm Overview

1. **LTL to Parity Automaton**: The LTL formula is translated into a parity automaton using spot. The automaton is completed and edges are split to facilitate matching with observations.

2. **Product POMDP Construction**: A product POMDP is constructed by combining the original POMDP with the parity automaton. States in the product are pairs (POMDP state, automaton state), and transitions synchronize the POMDP actions with automaton transitions based on the atomic propositions satisfied by each POMDP observation. Product observations also record the next automaton state, as required to preserve strong revelation under condition (2).

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

# Load and check a revealing POMDP
with open('examples/ltl-revealing/ltl-guard.pomdp', 'r') as f:
    pomdp = pomdp_parser.parse(f.read())

if is_strongly_revealing(pomdp):
    print("POMDP is strongly revealing")

# For labelled models, also check the paper's condition (2).
if not is_strongly_revealing(pomdp, preserve_labels=True):
    revealing_pomdp = make_strongly_revealing(pomdp)
```

The checker implements Definition 5.1: revealing observations identify the
destination across **all** source states for a fixed action. With
`preserve_labels=True`, every positive outcome must also have a revealing
alternative on the same edge with the same atomic propositions (condition (2)).
Both checks assume valid probability distributions.

### Transforming POMDPs to Strongly Revealing

For each outcome without a suitable revealing alternative, the transformation
uses a revealing observation with the same destination and atomic propositions.
An existing observation is reused if it identifies that destination across all
actions and has the required label. Otherwise, a new observation is created,
shared by outcomes with the same destination and label.
It splits the probability `p` into `(1 - epsilon) * p` on the original observation
and `epsilon * p` on the revealing observation (`epsilon=0.05` by default), adding
the transferred probabilities when observations are shared. For `AtomicPropPOMDP`,
this ensures condition (2), even if the input already satisfies ordinary strong
revelation. Existing revealing observations are retained.

The input model and source files are not modified. Joint next-state/observation
distributions are checked for finite, nonnegative probabilities summing to one
within `1e-12`. Invalid inputs are rejected without adjusting their probabilities.
For models specified using separate `T:` and `O:` tables, `T` gives the
probabilities of next states given the current state and action, while `O` gives the probabilities of
observations given the action and next state. Their product gives the joint
probability of the next state and observation, which is checked as described above.
Valid inputs keep the same state dynamics and label distributions; priorities and initial probabilities are
always preserved. The total probability of revelation can differ from `epsilon`,
since existing revelations are retained. Export uses 14 significant digits,
with scientific notation for small probabilities, so positive entries stay positive.
Set `epsilon` with `make_strongly_revealing(pomdp, epsilon=...)` or the CLI's
`--epsilon` option. It must lie strictly between zero and one.

**Using the revealing module directly:**

```bash
# Check if a POMDP is revealing
python -m pomdpy.revealing --file examples/ltl-revealing/ltl-guard.pomdp

# Transform a non-revealing POMDP to be strongly revealing
python -m pomdpy.revealing --file examples/ltl/ltl-guard.pomdp --transform

# Transform and save to a specific file
python -m pomdpy.revealing --file examples/ltl/ltl-guard.pomdp --transform --output examples/ltl-revealing/ltl-guard-transformed.pomdp

# Transform without checking first (faster for large files)
python -m pomdpy.revealing --file examples/ltl/ltl-guard.pomdp --transform --no-check
```

The module automatically:
- Preserves the POMDP type (regular, AtomicPropPOMDP, or ParityPOMDP)
- Preserves atomic proposition information for LTL specifications
- Verifies the transformation succeeded
- Rejects invalid transition distributions
- Skips transformation if the POMDP already has suitable revealing alternatives, including matching labels for AtomicPropPOMDP

**Batch transformation script:**

```bash
python scripts/transform_to_revealing.py
```

This script:
- Processes POMDP files from source directories (`examples/ltl/`, `examples/parity/`, etc.)
- Checks strong revelation and, for labelled models, condition (2)
- Validates transition distributions, including for already revealing models
- Adds revealing observations where either requirement is missing
- Saves transformed files to corresponding revealing directories (`ltl-revealing/`, `parity-revealing/`)
- Copies matching TLSF files alongside transformed POMDPs
- Applies a 60-second timeout per file
- Preserves atomic proposition information for POMDPs with LTL specifications

### Benchmark Sources

The parity benchmarks are adapted from the PPS (Parity POMDP Solver) benchmark
suite accompanying Krishnendu Chatterjee, Martin Chmelík, Raghav Gupta, and Ayush
Kanodia, [*Qualitative analysis of POMDPs with temporal logic specifications for
robotics applications*](https://doi.org/10.1109/ICRA.2015.7139019), ICRA 2015,
pp. 325–330. The original tool website was `http://pub.ist.ac.at/pps/`; an
[archived copy](https://web.archive.org/web/20191212064315/http://pub.ist.ac.at/pps/index.html)
and the bundled [PPS manual](examples/pps/pps.pdf) provide further information.

The PPS files in `examples/pps/` already contain parity priorities and transition
supports. [The converter](scripts/convert_pps_to_pomdp.py) produces the files in
`examples/parity/`, assigning uniform probabilities to successor supports.
[The revealing transformation](scripts/transform_to_revealing.py) then produces
`examples/parity-revealing/` by adding revealing observations where needed.

The Tiger LTL example is adapted from the
[original discounted Tiger benchmark](https://www.pomdp.org/examples/tiger.aaai.POMDP),
with explicit success/death states, a restart action, and revealing observations.
The LTL models and specifications are in `examples/ltl/`, with transformed
versions in `examples/ltl-revealing/`. The Corridor and Guard examples are newly
created for this repository.

### Benchmarking

Run comprehensive benchmarks on all revealing instances:

```bash
python scripts/run_benchmark_revealing.py
```

This script:
- Tests revealing POMDP files from `examples/ltl-revealing/` and `examples/parity-revealing/` directories
- Records detailed metrics for each instance:
  - POMDP sizes (states, actions, observations)
  - Automaton sizes (states, edges)
  - Belief-support MDP size
  - Phase-by-phase timing (automaton construction, belief-support construction, solving)
  - Whether the instance is strongly revealing
  - Whether the initial belief admits an almost-sure winning strategy (`initial_winning`: `yes`/`no`, blank for timeouts or errors)
- Outputs results to `examples/benchmark_revealing.csv`
- Uses a 5-minute timeout per instance

## Simulating POMDPs

You can use `pomdpy/sim.py` to simulate a given POMDP. You will have to choose
actions to step through the environment and you will only get as feedback the
(support) of the beliefs. Internally, the simulator will use a pseudo-random
number generator to select a concrete successor state.
