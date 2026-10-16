# POMDPs with Reveals

Python algorithms for POMDPs that reveal their state from time to time, with LTL
or parity objectives. LTL translation uses [Spot](https://spot.lre.epita.fr/).

## Installation

CI uses Python 3.12 on Ubuntu 24.04. Install Graphviz and its development headers
before installing the Python dependencies, since `pygraphviz` needs them.
The full test suite, belief-support algorithm, and benchmark runner also require
Spot's Python bindings, which `requirements.txt` does not install.

On Ubuntu 24.04, the [CI workflow](.github/workflows/python-app.yml) shows how to
add Spot's official OBS package repository and install
`python3-spot=2.16.1.0-1` (Spot 2.16.1), along with `graphviz` and `graphviz-dev`.
This uses a prebuilt package and avoids compiling Spot from source.
Spot's bindings must match the Python version used to run the programs.

With those system dependencies installed, [create a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
and install the Python requirements from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Spot must also be visible inside the environment. When using the Ubuntu package
with a virtual environment, prefix Spot-dependent commands with
`PYTHONPATH=/usr/lib/python3/dist-packages`, as CI does for its test steps.
For example, run the full suite with
`PYTHONPATH=/usr/lib/python3/dist-packages python -m pytest`.

The algorithm CLI and batch transformation script use `SIGALRM` and require a
Unix-like platform.

## Regenerating benchmarks

With the dependencies installed, run these commands from the repository root:

```bash
python scripts/convert_pps_to_pomdp.py --dir examples/pps --output examples/parity
python scripts/transform_to_revealing.py
```

The first command converts the PPS inputs into `examples/parity/`. The second
generates revealing models in `examples/parity-revealing/` and
`examples/ltl-revealing/`. Existing output files are overwritten; the original
PPS inputs are preserved. To rerun the experiments, see [Benchmarking](#benchmarking).

## Running tests

With Python 3.12, the dependencies, and Spot's Python bindings installed, run
from the repository root:

```bash
python scripts/transform_to_revealing.py
python -m pytest
```

The first command generates the models required by the regression tests; skip it
if they have already been generated.

The test command also measures line and branch coverage using `pytest-cov`,
included in `requirements.txt`. The combined coverage must reach 79.5%, as
configured in `.coveragerc`; a lower result fails the command.

Coverage includes the library, `belief_support_algorithm.py`, and the benchmark
and conversion scripts, including code run in subprocesses. Tests, third-party
code, and the test snapshot generator `scripts/regenerate_expected_results.py`
are excluded. Open `htmlcov/index.html` for the detailed report; generated
coverage files are ignored by Git. The terminal's `Cover` column combines line
and branch coverage; the JSON report contains their separate counts.

Run the same Flake8 check as CI with:

```bash
python -m flake8 belief_support_algorithm.py pomdpy scripts
```

`setup.cfg` defines the style rules, including the exceptions for existing
formatting. For pull requests targeting `main` and pushes to `main`, CI checks
style, generates the test models, and runs the full test suite.

## Belief-support algorithm

`belief_support_algorithm.py` computes almost-sure winning states and strategies
for strongly revealing POMDPs. It accepts an `AtomicPropPOMDP` with an LTL
objective, or a `ParityPOMDP` with priorities already defined. Parity inputs skip
LTL translation and product construction, so they need no formula or TLSF file.

### Usage

With an LTL formula:

```bash
python belief_support_algorithm.py --file <pomdp-file> --ltl_formula "<formula>" [options]
```

With a TLSF file:

```bash
python belief_support_algorithm.py --file <pomdp-file> --tlsf_file <tlsf-file> [options]
```

The TLSF (Temporal Logic Synthesis Format) option reads the `INPUTS` and
`GUARANTEES` sections used by the bundled specifications. Multiple guarantees
are combined with conjunction, and TLSF guarantees take precedence over
`--ltl_formula`. The CLI uses all atomic propositions defined in the POMDP unless
`--atoms` is supplied. See the `examples/` directory for paired POMDP and TLSF files.

Options:

- `--verbose`: Detailed step-by-step output showing each phase of the algorithm
- `--plot`: Write DOT files and render PNGs with `pygraphviz` and Graphviz
- `--atoms`: Manually specify atomic proposition IDs (usually auto-detected)
- `--output_dir`: Directory for visualizations (default: `figs`)

The CLI stops a computation after 30 seconds. For longer experiments, use the
[benchmark runner](#benchmarking), which allows five minutes per instance.

Examples:

```bash
# Using TLSF file (auto-detects formula and atoms)
python belief_support_algorithm.py --file examples/ltl-revealing/ltl-guard.pomdp \
    --tlsf_file examples/ltl-revealing/ltl-guard.tlsf --verbose --plot

# Using explicit formula
python belief_support_algorithm.py --file examples/ltl-revealing/ltl-guard.pomdp \
    --ltl_formula "Fp0" --atoms 0,1 --verbose
```

### Algorithm overview

1. LTL translation: Spot translates the formula into a complete, deterministic parity automaton with state-based acceptance. Edges are split to match observations.

2. Product construction: The original POMDP is combined with the parity automaton. Product states are pairs (POMDP state, automaton state), and transitions synchronize POMDP actions with automaton transitions based on each observation's atomic propositions. Product observations also record the next automaton state, as required to preserve strong revelation under condition (2).

3. Belief-support construction: Forward exploration starts from the initial belief support. Each MDP state is a set of product states, or POMDP states for direct parity inputs. Transitions follow:

   - Taking an action in the POMDP
   - Observing the resulting observation
   - Updating the belief support and, for LTL inputs, the automaton state

4. Parity solving: The solver computes almost-sure winning states and strategies using:

   - Maximal End Component (MEC) decomposition to find strongly connected sets closed under selected actions
   - Filtering MECs by their maximum priority
   - Computing almost-sure reachability to good MECs
   - Extracting strategies for reaching and staying within winning regions

### Output

The algorithm reports:

- Almost-sure winning POMDP states derived from winning singleton belief supports
- Number of winning belief-support states; `--verbose` also reports the total explored
- Whether the input POMDP is strongly revealing

With `--plot`, LTL inputs produce `automaton.dot` and `product_pomdp.dot`.
All supported inputs produce `belief_support_mdp.dot`, with reachability
strategies in blue and MEC strategies in green. When `pygraphviz` and Graphviz
are available, the automaton and belief-support plots are also rendered as PNGs.
The product is saved as DOT only. Files go to `figs/` unless `--output_dir` is set.

### Checking if a POMDP is revealing

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

### Transforming POMDPs to strongly revealing

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

Using the revealing module directly:

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

Batch transformation script:

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

### Benchmark sources

The parity benchmarks are adapted from the PPS (Parity POMDP Solver) benchmark
suite accompanying Krishnendu Chatterjee, Martin Chmelík, Raghav Gupta, and Ayush
Kanodia, [*Qualitative analysis of POMDPs with temporal logic specifications for
robotics applications*](https://doi.org/10.1109/ICRA.2015.7139019), ICRA 2015,
pp. 325-330. The original tool website was `http://pub.ist.ac.at/pps/`; an
[archived copy](https://web.archive.org/web/20191212064315/http://pub.ist.ac.at/pps/index.html)
and the bundled [PPS manual](examples/pps/pps.pdf) provide further information.

The PPS files in `examples/pps/` already contain parity priorities and transition
supports. [The converter](scripts/convert_pps_to_pomdp.py) produces the files in
`examples/parity/`, assigning uniform probabilities to successor supports.
Repeated observation names are kept only once, in their original order; the PPS
source files are unchanged.
[The revealing transformation](scripts/transform_to_revealing.py) then produces
`examples/parity-revealing/` by adding revealing observations where needed.

The Tiger LTL example is adapted from the
[original discounted Tiger benchmark](https://www.pomdp.org/examples/tiger.aaai.POMDP),
with explicit success/death states, a restart action, and revealing observations.
The LTL models and specifications are in `examples/ltl/`, with transformed
versions in `examples/ltl-revealing/`. The Corridor and Guard examples are newly
created for this repository.

### Benchmarking

After [regenerating the benchmarks](#regenerating-benchmarks), run all revealing
instances:

```bash
python scripts/run_benchmark_revealing.py
```

Each instance runs in a separate process, with a 5-minute timeout and a
16 GiB memory limit by default. To change the memory limit, for example to 8 GiB:

```bash
python scripts/run_benchmark_revealing.py --memory-limit-gb 8
```

The memory limit requires Python's Unix-only `resource` module and the operating
system's `RLIMIT_AS` facility; it has been tested on Linux. It applies to virtual
memory, including loaded libraries, rather than only resident RAM. One GiB is
2^30 bytes; any stricter inherited limit still
applies.

To run without adding memory limits, including on systems without `resource`
or `RLIMIT_AS`, use:

```bash
python scripts/run_benchmark_revealing.py --no-memory-limit
```

This mode does not import `resource` or set resource limits. Process isolation
and the 5-minute timeout remain active, and `memory_limit_gb` is recorded as
`none` in the CSV. Existing system limits still apply. The two memory options
are mutually exclusive; unsupported limits are never silently disabled.

The parent process enforces the timeout, including when a native library is
unresponsive. Timeouts, memory allocation failures (`memory_limit` with a
configured limit, otherwise `error: out of memory`), and other worker errors
are recorded in the CSV, and the script continues with the next
instance.

This script:

- Tests revealing POMDP files from `examples/ltl-revealing/` and `examples/parity-revealing/` directories
- Records detailed metrics for each instance:
  - Instance path relative to `examples/`, retaining subdirectories to distinguish variants with the same filename
  - POMDP sizes (states, actions, observations)
  - Automaton sizes (states, edges)
  - Belief-support MDP size
  - Phase-by-phase timing (automaton construction, belief-support construction, solving)
  - Whether the instance is strongly revealing
  - Whether the initial belief admits an almost-sure winning strategy (`initial_winning`: `yes`/`no`, blank for failed instances)
  - The configured limits (`memory_limit_gb`, `timeout_seconds`)
- Outputs results to `examples/benchmark_revealing.csv`.

## Simulating POMDPs

Use `pomdpy/sim.py` to step through a POMDP interactively. Run it from the
repository root as a module:

```bash
python -m pomdpy.sim examples/ltl/ltl-guard.pomdp
```

Choose an action at each step, or press Enter to repeat the previous action.
The simulator samples a concrete successor state and displays the observation,
the sampled state, and the updated belief probabilities. Probabilities below
0.001 are omitted from the display. Press Ctrl+C to stop.
