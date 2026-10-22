"""Convert TLSF specifications using the tlsf-tools executables on PATH."""

from pathlib import Path
import re
import subprocess

import spot


def _run_tlsf_tool(tool, *args):
    try:
        result = subprocess.run(
            [tool, *args], check=True, capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{tool} not found on PATH; install tlsf-tools as described in README.md"
        ) from error
    except subprocess.CalledProcessError as error:
        diagnostic = error.stderr.strip() or f"exit status {error.returncode}"
        raise ValueError(f"{tool} failed: {diagnostic}") from error
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"{tool} exceeded its 30-second timeout") from error
    return result.stdout.strip()


def parse_tlsf_file(tlsf_path):
    """Return the full LTL 'formula' and declared 'inputs' as proposition indices."""
    # Resolve first to distinguish a missing input from a missing executable,
    # and to keep filenames beginning with '-' from being read as options.
    path = Path(tlsf_path).resolve(strict=True)
    semantics = _run_tlsf_tool('tlsfinfo', '--semantics', path)
    if 'Finite' in semantics:
        raise ValueError("Finite-word TLSF semantics are not supported by the parity solver")

    inputs = _run_tlsf_tool('tlsfinfo', '--expanded-ins', path)
    inputs = {name.strip() for name in inputs.split(',') if name.strip()}
    # The default ltlxba format lowercases propositions; keep their names intact.
    formula = _run_tlsf_tool('tlsf2ltl', '--format', 'ltl', path)
    if not formula:
        raise ValueError("tlsf2ltl returned an empty formula")

    # Declarations and propositions used in the objective must use POMDP names.
    # Unused outputs (such as the examples' dummy 'action') need no mapping.
    names = inputs | {ap.ap_name() for ap in spot.atomic_prop_collect(spot.formula(formula))}
    invalid = sorted(name for name in names if not re.fullmatch(r'p(?:0|[1-9][0-9]*)', name))
    if invalid:
        raise ValueError(f"TLSF propositions must be named p0, p1, ...; unsupported: {', '.join(invalid)}")
    return {'formula': formula, 'inputs': sorted(int(name[1:]) for name in inputs)}
