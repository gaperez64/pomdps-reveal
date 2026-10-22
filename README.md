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

TLSF input also requires `tlsf2ltl` and `tlsfinfo` from
[tlsf-tools](https://github.com/gaperez64/tlsf-tools) on `PATH`. These are external
executables, not Python packages. With the virtual environment active, install
the tested v1.8 release on Linux x86-64 as follows:

```bash
tlsf_dir=$(mktemp -d)
curl -fL \
  https://github.com/gaperez64/tlsf-tools/releases/download/v1.8/tlsf-tools-v1.8-linux-x86_64-v2.tar.gz \
  -o "$tlsf_dir/tlsf.tar.gz"
printf '%s  %s\n' \
  a178c77dba23ba6163f0ebf85dc2fdcb098d28475a124be9039efe7f5af254bf \
  "$tlsf_dir/tlsf.tar.gz" | sha256sum -c -
tar -xzf "$tlsf_dir/tlsf.tar.gz" -C "$tlsf_dir"
install -m 755 \
  "$tlsf_dir/tlsf-tools-v1.8-linux-x86_64-v2/tlsf2ltl" \
  "$tlsf_dir/tlsf-tools-v1.8-linux-x86_64-v2/tlsfinfo" \
  "${VIRTUAL_ENV:?Activate the virtual environment first}/bin/"
tlsf2ltl --version
tlsfinfo --version
```

The [release page](https://github.com/gaperez64/tlsf-tools/releases/tag/v1.8)
also provides macOS ARM binaries. For other platforms, follow the upstream
build instructions. Parity inputs and direct LTL formulas do not require
tlsf-tools.

The algorithm CLI and batch transformation script use `SIGALRM` and require a
Unix-like platform.

## Regenerating benchmarks

To reproduce the published corpus, start with empty `examples/parity/`,
`examples/parity-revealing/`, and `examples/ltl-revealing/` directories.
A failed or timed-out transformation can otherwise leave an older output file
in place.

With the dependencies installed, run these commands from the repository root:

```bash
python scripts/convert_pps_to_pomdp.py --dir examples/pps --output examples/parity
python scripts/transform_to_revealing.py
```

The first command converts the PPS inputs into `examples/parity/` and translates
their priorities from minimum-even to maximum-even. The second
generates revealing models in `examples/parity-revealing/` and
`examples/ltl-revealing/`. Existing output files are overwritten; the original
PPS inputs are preserved. To rerun the experiments, see [Benchmarking](#benchmarking).

## Running tests

With Python 3.12, the dependencies, Spot's Python bindings, and tlsf-tools
installed, run from the repository root:

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
objective and label-preserving revealing alternatives, or a `ParityPOMDP`
with priorities already defined. Parity inputs skip
LTL translation and product construction, so they need no formula or TLSF file.
They use maximum-even acceptance: the highest priority seen infinitely often
must be even.

### Usage

With an LTL formula:

```bash
python belief_support_algorithm.py --file <pomdp-file> --ltl_formula "<formula>" [options]
```

With a TLSF file:

```bash
python belief_support_algorithm.py --file <pomdp-file> --tlsf_file <tlsf-file> [options]
```

The TLSF (Temporal Logic Synthesis Format) option uses tlsf-tools to convert the
whole specification into an LTL objective, including assumptions, guarantees,
and initial conditions. Definitions and parameters are expanded by tlsf-tools.
The resulting formula takes precedence over `--ltl_formula`. Finite-word
specifications are rejected because the solver handles infinite executions.
Declared inputs and propositions used in the formula must use the POMDP's
`p0`, `p1`, ... names; unused outputs such as the examples' `action` are allowed.
Each call to tlsf-tools has a 30-second timeout. The CLI uses all atomic
propositions defined in the POMDP unless `--atoms` is supplied.
See the `examples/` directory for paired POMDP and TLSF files.

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

- Whether the initial belief admits an almost-sure winning strategy
- POMDP states occurring in reachable winning singleton belief supports
- Number of winning belief-support states; `--verbose` also reports the total explored
- Whether the input POMDP is strongly revealing, including label preservation for LTL inputs

For LTL, the state list omits the automaton state, which records past observations.
A listed state may therefore lose when starting afresh with the same objective.
The initial-belief answer accounts for the initial automaton state.

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
- Applies a 5-minute timeout per file
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
PPS uses minimum-even parity, whereas this implementation uses maximum-even
parity. The converter replaces each priority `p` with `D - p`, where `D` is the
smallest even integer at least as large as the largest original priority.
For example, priorities `1, 2, 3` become `3, 2, 1`. This preserves the objective
and is done before constructing belief supports.
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

Each LTL instance requires a matching `.tlsf` file next to its `.pomdp` file,
copied automatically during regeneration. A missing specification is recorded
as an error.

Each instance runs five times by default, sequentially, with a fresh process
for every repetition. Each run has a 5-minute timeout and a 16 GiB memory limit
by default. Set `--repetitions N` to change the number of runs.
For five repetitions with a 24 GiB limit, use:

```bash
python scripts/run_benchmark_revealing.py --repetitions 5 --memory-limit-gb 24
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
are recorded in the individual-run CSV, and the script continues with the next
repetition or instance.

This script:

- Tests revealing POMDP files from `examples/ltl-revealing/` and `examples/parity-revealing/` directories
- Records detailed metrics for each instance:
  - Instance path relative to `examples/`, retaining subdirectories to distinguish variants with the same filename
  - POMDP sizes (states, actions, observations)
  - Automaton sizes (states, edges)
  - Belief-support MDP size
  - Phase-by-phase timing (automaton construction, belief-support construction, solving)
  - Whether the instance is strongly revealing, including label preservation for LTL inputs (`revealing`)
  - Whether the initial belief admits an almost-sure winning strategy (`initial_winning`: `yes`/`no`, blank for failed instances)
  - The configured limits (`memory_limit_gb`, `timeout_seconds`)
- Saves every repetition to `examples/benchmark_revealing_runs.csv`, with a
  `run` column numbered from 1 for each instance.
- Saves one summary row per instance to `examples/benchmark_revealing.csv`,
  including `repetitions`, `successful_runs`, `timeout_runs`, `memory_limit_runs`,
  and `error_runs`. The existing `t_aut`, `t_bs`, and `t_solve` columns contain
  medians. Each phase also has columns with suffixes `_mean`, `_stddev`, `_min`,
  and `_max`. `_stddev` is the sample standard deviation (denominator `n - 1`)
  and is blank for a single repetition. All timing statistics are reported only
  if all repetitions succeed and agree on the sizes and qualitative results.
  Otherwise timing statistics and model results are blank; differing statuses
  are marked `mixed`, and conflicting
  successful results are flagged as an error. The raw CSV retains all details.

The phase times measure elapsed wall-clock time. For LTL inputs, `t_aut` includes
Spot translation and edge splitting, and `t_bs` includes both product construction
and belief-support construction. For parity inputs, `t_aut` is zero and `t_bs`
measures belief-support construction alone. `t_solve` measures parity solving.
These phase times exclude process startup, input parsing, and the separate check
that the input POMDP is strongly revealing with matching labels for LTL inputs;
the 5-minute timeout covers the entire worker process.

### JAIR results

The measurements for the accepted JAIR version of *Revelations: A Decidable
Class of POMDPs with Omega-Regular Objectives* are archived in
[the summary CSV](results/jair/benchmark_revealing.csv) and
[the individual-run CSV](results/jair/benchmark_revealing_runs.csv).
The corresponding code revision is `157b2edaacec383125d6b9d1f506bdb67516bfe1`.
The evaluation used the default revelation parameter `epsilon=0.05`, five runs
per instance, a 300-second timeout, and a 24 GiB virtual-memory limit, using the
24 GiB command in [Benchmarking](#benchmarking). Hardware and software versions
are given in the paper's experimental setup (Section 6.2).

The evaluated corpus contains four LTL instances and 64 parity instances.
Two converted parity inputs, `RockSample/RS4_3_3.pomdp` and
`RockSample/RS4_3_4.pomdp`, had no generated revealing output and were excluded
from the evaluation. These generation exclusions are separate from the
memory-limit failures recorded in the CSVs.

After regenerating the models, reproduce the paper's LaTeX tables from the
archived summary CSV with:

```bash
python results/jair/generate_benchmark_tables.py
```

The script uses only the Python standard library and writes
`results/jair/tables/ltl_results.tex` and `results/jair/tables/parity_results.tex`.
It reads the generated models in `examples/` to recover input sizes missing
from failed-run CSV rows. Use `--examples-dir` or `--output-dir` to override
these directories; an alternative CSV can be passed as a positional argument.

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
