#!/usr/bin/env python3
"""
Run benchmarks on all strongly revealing POMDP instances and record phase runtimes and sizes.

Processes POMDP files from ltl-revealing and parity-revealing directories.
Outputs a CSV with one row per instance containing:
- instance path relative to examples/
- POMDP type (atomic or parity)
- POMDP sizes (states, actions, observations)
- automaton sizes (states, edges) [N/A for parity]
- belief-support MDP size (states)
- phase runtimes (automaton, belief-support construction, solver)
- status (ok/timeout/memory_limit/error)
- initial_winning (yes/no; blank for failed instances)
- memory limit and timeout

Timeout per instance: 300 seconds (5 minutes).
Memory limit per instance: 16 GiB of virtual memory (configurable or disabled).
"""

import argparse
import os
import sys
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

TIMEOUT_SECONDS = 300
MEMORY_ERROR_EXIT_CODE = 3


def find_tlsf_for_pomdp(pomdp_path: Path) -> Path | None:
    """Return matching TLSF file for a given revealing POMDP file, if present."""
    tlsf_path = pomdp_path.with_suffix(".tlsf")
    return tlsf_path if tlsf_path.exists() else None


def aut_sizes(aut) -> tuple[int, int]:
    """Return (num_states, num_edges) for a spot automaton."""
    try:
        return aut.num_states(), aut.num_edges()
    except MemoryError:
        raise
    except Exception:
        # Fallback: approximate via transitions iteration
        try:
            ns = len(list(aut.states()))
            ne = sum(len(list(aut.out(s))) for s in aut.states())
            return ns, ne
        except MemoryError:
            raise
        except Exception:
            return -1, -1


def run_instance(pomdp_file: Path) -> dict:
    """Run all phases for a single instance and return results dict."""
    # Import the solver libraries only in the worker, after setting its limits.
    from pomdpy.parsers import pomdp as pomdp_parser
    from pomdpy.pomdp import AtomicPropPOMDP, ParityPOMDP
    from pomdpy.product import ProductPOMDP
    from pomdpy.belief_support_MDP import BeliefSuppMDP
    from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
    from pomdpy.revealing import is_strongly_revealing
    import spot

    instance_name = pomdp_file.name
    result = {
        "instance": instance_name,
        "status": "ok",
        "pomdp_type": None,
        "pomdp_states": None,
        "pomdp_actions": None,
        "pomdp_observations": None,
        "aut_states": None,
        "aut_edges": None,
        "bs_states": None,
        "t_aut": None,
        "t_bs": None,
        "t_solve": None,
        "initial_winning": None,
    }

    # Load POMDP
    with open(pomdp_file, "r") as f:
        env = pomdp_parser.parse(f.read())
    result["pomdp_states"] = len(env.states)
    result["pomdp_actions"] = len(env.actions)
    result["pomdp_observations"] = len(env.obs)

    # Detect POMDP type
    is_parity = isinstance(env, ParityPOMDP)
    is_atomic = isinstance(env, AtomicPropPOMDP)
    result["pomdp_type"] = "parity" if is_parity else ("atomic" if is_atomic else "unknown")

    # Handle automaton construction based on POMDP type
    if is_parity:
        # ParityPOMDP: No automaton, no product
        # Build belief-support MDP directly from parity POMDP
        result["t_aut"] = 0.0
        result["aut_states"] = None  # N/A for parity
        result["aut_edges"] = None   # N/A for parity

        # Construct belief-support MDP for ParityPOMDP (aut=None)
        t2 = time.perf_counter()
        bsmdp = BeliefSuppMDP(env, aut=None)
        t3 = time.perf_counter()
        result["t_bs"] = t3 - t2
        result["bs_states"] = len(bsmdp.states)

        # Solve parity MDP
        t4 = time.perf_counter()
        solver = ParityMDPSolver(bsmdp, verbose=False)
        max_prio = max(bsmdp.prio.values()) if bsmdp.prio else 0
        aswin, _, _ = solver.almostSureWin(max_priority=max_prio)
        t5 = time.perf_counter()
        result["t_solve"] = t5 - t4
    else:
        # AtomicPropPOMDP: Build automaton from TLSF if available
        tlsf_path = find_tlsf_for_pomdp(pomdp_file)
        if tlsf_path is None:
            # Minimal safe formula if no TLSF
            formula = "G F p0" if hasattr(env, "atoms") and env.atoms else "true"
            t0 = time.perf_counter()
            aut = spot.translate(
                formula, "parity max even", "deterministic", "complete", "SBAcc", "colored"
            )
            aut = spot.split_edges(aut)
            t1 = time.perf_counter()
        else:
            # Derive formula from TLSF inputs/guarantees
            with open(tlsf_path, "r") as tf:
                tlsf = tf.read()
            # Extract GUARANTEES naive (lines ending with ';')
            import re
            m = re.search(r"GUARANTEES\s*\{([^}]+)\}", tlsf, re.DOTALL)
            formulas = []
            if m:
                for line in m.group(1).splitlines():
                    line = line.strip()
                    if line and not line.startswith("//") and line.endswith(";"):
                        line = line[:-1]
                        line = (
                            line.replace("&&", "&")
                            .replace("||", "|")
                            .replace("\\!", "!")
                            .replace("\\&", "&")
                            .replace("\\|", "|")
                        )
                        formulas.append(line)
            formula = " & ".join(f"({f})" for f in formulas) if formulas else "true"
            t0 = time.perf_counter()
            aut = spot.translate(
                formula, "parity max even", "deterministic", "complete", "SBAcc", "colored"
            )
            aut = spot.split_edges(aut)
            t1 = time.perf_counter()

        result["t_aut"] = t1 - t0
        a_states, a_edges = aut_sizes(aut)
        result["aut_states"] = a_states
        result["aut_edges"] = a_edges

        # Product + Belief-Support MDP for AtomicPropPOMDP
        t2 = time.perf_counter()
        prod = ProductPOMDP(env, aut)
        bsmdp = BeliefSuppMDP(prod, aut)
        t3 = time.perf_counter()
        result["t_bs"] = t3 - t2
        result["bs_states"] = len(bsmdp.states)

        # Solve parity MDP
        t4 = time.perf_counter()
        solver = ParityMDPSolver(bsmdp, verbose=False)
        max_prio = max(bsmdp.prio.values()) if bsmdp.prio else 0
        aswin, _, _ = solver.almostSureWin(max_priority=max_prio)
        t5 = time.perf_counter()
        result["t_solve"] = t5 - t4

    result["initial_winning"] = "yes" if bsmdp.start in aswin else "no"

    # Mark revealing status (informational)
    try:
        result["revealing"] = "yes" if is_strongly_revealing(env) else "no"
    except MemoryError:
        raise
    except Exception:
        result["revealing"] = "unknown"

    return result


def run_worker(pomdp_file: Path, result_file: Path, memory_limit_gb: int | None) -> int:
    """Apply any requested resource limits before loading and solving an instance."""
    try:
        try:
            if memory_limit_gb is not None:
                import resource

                limit = memory_limit_gb * 1024 ** 3
                try:
                    # Retain any stricter inherited limits.
                    for inherited in resource.getrlimit(resource.RLIMIT_AS):
                        if inherited != resource.RLIM_INFINITY:
                            limit = min(limit, inherited)
                    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                    # A native-library crash must not produce a large core dump.
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                except (OSError, ValueError) as e:
                    raise RuntimeError(
                        f"cannot set resource limits: {e}; "
                        "use --no-memory-limit to run without added limits"
                    ) from e
            row = run_instance(pomdp_file)
        except MemoryError:
            raise
        except Exception as e:
            row = {"instance": pomdp_file.name, "status": f"error: {e}"}
        result_file.write_text(json.dumps(row))
    except MemoryError:
        # No result needs to be allocated or serialized when memory is exhausted.
        return MEMORY_ERROR_EXIT_CODE
    return 0


def run_limited_instance(pomdp_file: Path, memory_limit_gb: int | None = 16,
                         timeout: float = TIMEOUT_SECONDS) -> dict:
    """Run one isolated worker, enforcing its wall-clock timeout from the parent."""
    row = {"instance": pomdp_file.name}
    memory_args = (["--no-memory-limit"] if memory_limit_gb is None else
                   ["--memory-limit-gb", str(memory_limit_gb)])
    with tempfile.TemporaryDirectory(prefix="pomdp-benchmark-") as tmp:
        result_file = Path(tmp) / "result.json"
        try:
            worker = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 *memory_args,
                 "--worker", str(pomdp_file), str(result_file)],
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        except OSError as e:
            row["status"] = f"error: cannot start worker: {e}"
        else:
            if worker.returncode == MEMORY_ERROR_EXIT_CODE:
                row["status"] = ("memory_limit" if memory_limit_gb is not None else
                                 "error: out of memory")
            elif worker.returncode != 0:
                # A native crash or external kill is not necessarily an OOM.
                row["status"] = f"error: worker exited with code {worker.returncode}"
            else:
                try:
                    row = json.loads(result_file.read_text())
                except (OSError, ValueError) as e:
                    row["status"] = f"error: cannot read worker result: {e}"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    memory_options = parser.add_mutually_exclusive_group()
    memory_options.add_argument(
        "--memory-limit-gb", type=int, default=16,
        help="virtual-memory limit per instance in GiB (default: 16)",
    )
    memory_options.add_argument(
        "--no-memory-limit", dest="memory_limit_gb", action="store_const", const=None,
        help="do not add a memory limit (does not require the resource module)",
    )
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_limit_gb is not None:
        if not 0 < args.memory_limit_gb <= sys.maxsize // 1024 ** 3:
            parser.error("--memory-limit-gb must be a positive integer within the system's address range")
        try:
            import resource
        except ImportError:
            parser.error("memory limits require the resource module; use --no-memory-limit to opt out")
        if not hasattr(resource, "RLIMIT_AS"):
            parser.error("memory limits require OS support for RLIMIT_AS; use --no-memory-limit to opt out")
    if args.worker:
        return run_worker(Path(args.worker[0]), Path(args.worker[1]), args.memory_limit_gb)

    examples_dir = Path(ROOT) / "examples"
    out_csv = Path(ROOT) / "examples" / "benchmark_revealing.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Collect POMDP files from ltl-revealing and parity-revealing folders
    ltl_revealing_dir = examples_dir / "ltl-revealing"
    parity_revealing_dir = examples_dir / "parity-revealing"

    pomdp_files = []
    if ltl_revealing_dir.exists():
        pomdp_files.extend(sorted(ltl_revealing_dir.glob("**/*.pomdp")))
    if parity_revealing_dir.exists():
        pomdp_files.extend(sorted(parity_revealing_dir.glob("**/*.pomdp")))

    pomdp_files = [p for p in pomdp_files if p.is_file()]

    print(f"Found {len(pomdp_files)} instances ({len([p for p in pomdp_files if 'ltl-revealing' in str(p)])} ltl-revealing, {len([p for p in pomdp_files if 'parity-revealing' in str(p)])} parity-revealing)")
    memory_label = (f"{args.memory_limit_gb} GiB virtual memory"
                    if args.memory_limit_gb is not None else "no added memory limit")
    print(f"Limits per instance: {memory_label}, {TIMEOUT_SECONDS}s")

    # Prepare CSV
    fieldnames = [
        "instance", "status", "pomdp_type",
        "pomdp_states", "pomdp_actions", "pomdp_observations",
        "aut_states", "aut_edges",
        "bs_states",
        "t_aut", "t_bs", "t_solve",
        "revealing", "initial_winning",
        "memory_limit_gb", "timeout_seconds",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for i, pomdp_path in enumerate(pomdp_files, start=1):
            instance_name = pomdp_path.relative_to(examples_dir).as_posix()
            print(f"[{i}/{len(pomdp_files)}] {instance_name}", flush=True)
            row = run_limited_instance(pomdp_path, args.memory_limit_gb,
                                       timeout=TIMEOUT_SECONDS)
            row["instance"] = instance_name
            row["memory_limit_gb"] = args.memory_limit_gb if args.memory_limit_gb is not None else "none"
            row["timeout_seconds"] = TIMEOUT_SECONDS
            if row["status"] == "timeout":
                print(f"  ⊘ Timeout ({TIMEOUT_SECONDS}s)")
            elif row["status"] == "memory_limit":
                print(f"  ⊘ Memory limit ({args.memory_limit_gb} GiB)")
            elif row["status"] != "ok":
                print(f"  ✗ {row['status']}")

            writer.writerow(row)
            f.flush()

    print(f"\nCSV written to: {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
