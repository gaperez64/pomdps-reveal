#!/usr/bin/env python3
"""
Run benchmarks on all revealing LTL POMDP instances and record phase runtimes and sizes.

Outputs a CSV with one row per instance containing:
- instance name
- POMDP sizes (states, actions, observations)
- automaton sizes (states, edges)
- belief-support MDP size (states)
- phase runtimes (automaton, belief-support construction, solver)
- status (ok/timeout/error)

Timeout per instance: 300 seconds (5 minutes).
"""

import os
import sys
import csv
import signal
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from pomdpy.parsers import pomdp as pomdp_parser  # noqa: E402
from pomdpy.product import ProductPOMDP  # noqa: E402
from pomdpy.belief_support_MDP import BeliefSuppMDP  # noqa: E402
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver  # noqa: E402
from pomdpy.revealing import is_strongly_revealing  # noqa: E402
import spot  # noqa: E402


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Instance timed out")


def find_tlsf_for_pomdp(pomdp_path: Path) -> Path | None:
    """Return matching TLSF file for a given revealing POMDP file, if present."""
    tlsf_path = pomdp_path.with_suffix(".tlsf")
    return tlsf_path if tlsf_path.exists() else None


def aut_sizes(aut) -> tuple[int, int]:
    """Return (num_states, num_edges) for a spot automaton."""
    try:
        return aut.num_states(), aut.num_edges()
    except Exception:
        # Fallback: approximate via transitions iteration
        try:
            ns = len(list(aut.states()))
            ne = sum(len(list(aut.out(s))) for s in aut.states())
            return ns, ne
        except Exception:
            return -1, -1


def run_instance(pomdp_file: Path) -> dict:
    """Run all phases for a single instance and return results dict."""
    instance_name = pomdp_file.name
    result = {
        "instance": instance_name,
        "status": "ok",
        "pomdp_states": None,
        "pomdp_actions": None,
        "pomdp_observations": None,
        "aut_states": None,
        "aut_edges": None,
        "bs_states": None,
        "t_aut": None,
        "t_bs": None,
        "t_solve": None,
    }

    # Load POMDP
    with open(pomdp_file, "r") as f:
        env = pomdp_parser.parse(f.read())
    result["pomdp_states"] = len(env.states)
    result["pomdp_actions"] = len(env.actions)
    result["pomdp_observations"] = len(env.obs)

    # Build automaton from TLSF if available, else use a minimal placeholder
    tlsf_path = find_tlsf_for_pomdp(pomdp_file)
    if tlsf_path is None:
        # Minimal safe formula (true) so automaton is trivial
        formula = "G F p0" if hasattr(env, "atoms") and env.atoms else "true"
        t0 = time.perf_counter()
        aut = spot.translate(formula, "parity", "complete", "SBAcc")
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
        aut = spot.translate(formula, "parity", "complete", "SBAcc")
        aut = spot.split_edges(aut)
        t1 = time.perf_counter()

    result["t_aut"] = t1 - t0
    a_states, a_edges = aut_sizes(aut)
    result["aut_states"] = a_states
    result["aut_edges"] = a_edges

    # Product + Belief-Support MDP
    t2 = time.perf_counter()
    prod = ProductPOMDP(env, aut)
    bsmdp = BeliefSuppMDP(prod, aut)
    t3 = time.perf_counter()
    result["t_bs"] = t3 - t2
    result["bs_states"] = len(bsmdp.states)

    # Solve parity MDP
    solver = ParityMDPSolver(bsmdp, verbose=False)
    max_prio = max(bsmdp.prio.values()) if bsmdp.prio else 0
    t4 = time.perf_counter()
    _ = solver.almostSureWin(max_priority=max_prio)
    t5 = time.perf_counter()
    result["t_solve"] = t5 - t4

    # Mark revealing status (informational)
    try:
        result["revealing"] = "yes" if is_strongly_revealing(env) else "no"
    except Exception:
        result["revealing"] = "unknown"

    return result


def main():
    examples_dir = Path(ROOT) / "examples"
    out_csv = Path(ROOT) / "examples" / "benchmark_revealing.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pomdp_files = sorted(
        p for p in examples_dir.glob("revealing_ltl-*.pomdp")
        if p.is_file()
    )

    print(f"Found {len(pomdp_files)} revealing instances")

    # Prepare CSV
    fieldnames = [
        "instance", "status",
        "pomdp_states", "pomdp_actions", "pomdp_observations",
        "aut_states", "aut_edges",
        "bs_states",
        "t_aut", "t_bs", "t_solve",
        "revealing",
    ]

    # Set global timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, pomdp_path in enumerate(pomdp_files, start=1):
            print(f"[{i}/{len(pomdp_files)}] {pomdp_path.name}")
            signal.alarm(300)  # 5 minutes per instance
            try:
                row = run_instance(pomdp_path)
            except TimeoutError:
                row = {
                    "instance": pomdp_path.name,
                    "status": "timeout",
                    "pomdp_states": None,
                    "pomdp_actions": None,
                    "pomdp_observations": None,
                    "aut_states": None,
                    "aut_edges": None,
                    "bs_states": None,
                    "t_aut": None,
                    "t_bs": None,
                    "t_solve": None,
                    "revealing": None,
                }
                print("  ⊘ Timeout (300s)")
            except Exception as e:
                row = {
                    "instance": pomdp_path.name,
                    "status": f"error: {e}",
                    "pomdp_states": None,
                    "pomdp_actions": None,
                    "pomdp_observations": None,
                    "aut_states": None,
                    "aut_edges": None,
                    "bs_states": None,
                    "t_aut": None,
                    "t_bs": None,
                    "t_solve": None,
                    "revealing": None,
                }
                print(f"  ✗ Error: {e}")
            finally:
                signal.alarm(0)

            writer.writerow(row)

    print(f"\nCSV written to: {out_csv}")


if __name__ == "__main__":
    main()
