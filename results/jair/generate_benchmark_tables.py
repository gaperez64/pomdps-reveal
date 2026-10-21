#!/usr/bin/env python3
"""Generate the JAIR paper's LaTeX tables from the archived benchmark summary CSV."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
SIZE_COLUMNS = [
    ("pomdp_states", r"$|\sts|$"),
    ("pomdp_actions", r"$|\actions|$"),
    ("pomdp_observations", r"$|\signals|$"),
]
AUTOMATON_COLUMNS = [("aut_states", r"$|\atmtn|$"), ("aut_edges", "Edges")]
TIMING_COLUMNS = [("t_bs", r"$t_{bs}$ (s)"), ("t_solve", r"$t_{solve}$ (s)")]
COUNTS = ("successful_runs", "timeout_runs", "memory_limit_runs", "error_runs")
LTL_NAMES = {
    "ltl-corridor-difficult": "Corridor (Diff)",
    "ltl-corridor-easy": "Corridor (Easy)",
    "ltl-guard": "Security Guard",
    "ltl-tiger-repeating": "Tiger",
}


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def instance_name(instance):
    """Keep the paper's short names, including the distinguishing variants."""
    path = PurePosixPath(instance)
    name = LTL_NAMES.get(path.stem, path.stem)
    for directory in path.parts[:-1]:
        if directory.startswith("Cheese maze ") and directory.endswith(" states"):
            size = directory.split()[2]
            name = name.replace("cheese_maze", f"cheese_maze{size}", 1)
            name = name.replace("_problem_", "_prb_").replace("_medium_", "_med_")
    if "manyStartingStates" in path.parts:
        if name == "hallwatAut2":
            name = "hallwayAut2"
        name += " - many starts"
    return name


def format_time(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"invalid timing: {value!r}")
    if 0 < seconds < 0.0001:
        return "$<0.0001$"
    return f"{seconds:.4f}"


def read_model_sizes(path):
    """Read generated model declarations without loading transition matrices."""
    fields = {"states": "pomdp_states", "actions": "pomdp_actions",
              "observations": "pomdp_observations"}
    sizes = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            key, separator, value = line.split("#", 1)[0].partition(":")
            key = key.strip()
            if separator and key in fields:
                names = value.split()
                count = int(names[0]) if len(names) == 1 and names[0].isdigit() else len(names)
                if count < 1:
                    raise ValueError(f"empty {key} declaration in {path}")
                sizes[fields[key]] = str(count)
                if len(sizes) == len(fields):
                    return sizes
            if key in ("T", "O"):
                break
    raise ValueError(f"missing states, actions, or observations declaration in {path}")


def read_results(path, examples_dir=None):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "instance", "status", "initial_winning", "repetitions",
            "memory_limit_gb", "timeout_seconds", "bs_states", *COUNTS,
            *(key for key, _ in SIZE_COLUMNS + AUTOMATON_COLUMNS + TIMING_COLUMNS),
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing summary CSV columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("the summary CSV is empty")
    seen = set()
    for row in rows:
        instance = row["instance"]
        if not instance or instance in seen:
            raise ValueError(f"missing or duplicate instance identifier: {instance!r}")
        seen.add(instance)
        if PurePosixPath(instance).parts[0] not in ("ltl-revealing", "parity-revealing"):
            raise ValueError(f"cannot classify instance: {instance}")
        repetitions = int(row["repetitions"])
        counts = [int(row[key]) for key in COUNTS]
        if repetitions < 1 or min(counts) < 0 or sum(counts) != repetitions:
            raise ValueError(f"invalid repetition counts for {instance}")
        if not row["status"]:
            raise ValueError(f"missing status for {instance}")
        if row["status"] == "ok":
            if counts[0] != repetitions or row["initial_winning"] not in ("yes", "no"):
                raise ValueError(f"incomplete successful result for {instance}")
        elif any(not row[key] for key, _ in SIZE_COLUMNS):
            sizes = read_model_sizes((examples_dir or path.parent) / instance)
            for key, value in sizes.items():
                if row[key] and int(row[key]) != int(value):
                    raise ValueError(f"{key} differs between the CSV and model for {instance}")
                row[key] = value
    return sorted(rows, key=lambda row: row["instance"])


def caption(rows, kind):
    title = "LTL benchmarks" if kind == "ltl" else "parity-based benchmarks"
    text = f"Performance metrics for the {title}. "
    settings = {(row["repetitions"], row["timeout_seconds"], row["memory_limit_gb"])
                for row in rows}
    if len(settings) == 1:
        repetitions, timeout, memory = next(iter(settings))
        text += f"Each instance was run {int(repetitions)} times with a {int(timeout)}-second timeout"
        if memory != "none":
            text += f" and a {int(memory)}~GiB virtual-memory limit"
        else:
            text += " and no added memory limit"
        text += ". "
    else:
        text += "Repetition counts and resource limits are given in the source CSV. "
    text += ("Times are medians in seconds, reported only when all runs succeed and agree. "
             "`Win?' indicates whether the initial belief admits an almost-sure winning strategy. ")
    statuses = {row["status"] for row in rows}
    meanings = ["OK: all runs succeeded"] if "ok" in statuses else []
    if "memory_limit" in statuses:
        meanings.append("ML: all runs reached the memory limit")
    if "timeout" in statuses:
        meanings.append("TO: all runs timed out")
    if "mixed" in statuses:
        meanings.append("Mixed: runs had different statuses")
    if statuses - {"ok", "memory_limit", "timeout", "mixed"}:
        meanings.append("Error: worker error or inconsistent results")
    text += "; ".join(meanings) + "."
    if statuses != {"ok"}:
        text += " Dashes indicate unavailable synthesis results."
    return text


def render_table(rows, kind):
    columns = SIZE_COLUMNS + (AUTOMATON_COLUMNS if kind == "ltl" else [])
    columns += [("bs_states", r"$|\beliefMDP|$")] + TIMING_COLUMNS
    header = " & ".join([r"\textbf{Instance}"] + [label for _, label in columns]
                        + ["Win?", "Status"]) + r" \\"
    alignment = "@{}l" + "r" * len(columns) + "cc@{}"
    label = "tab:results" if kind == "ltl" else "tab:parity_results"
    names = [instance_name(row["instance"]) for row in rows]
    duplicates = {name for name, count in Counter(names).items() if count > 1}
    # Fall back to the full relative path if a future model makes a short name ambiguous.
    names = [str(PurePosixPath(row["instance"]).with_suffix("")) if name in duplicates else name
             for row, name in zip(rows, names)]
    if len(set(names)) != len(names):
        raise ValueError("the table's instance names are not unique")
    lines = ["% Generated by generate_benchmark_tables.py; edit the generator, not this file.",
             r"\begingroup", r"\small", r"\setlength{\tabcolsep}{3pt}"]
    if kind == "ltl":
        lines += [r"\begin{table}[ht]", r"\centering", r"\small",
                  rf"\caption{{{caption(rows, kind)}}}", rf"\label{{{label}}}",
                  rf"\begin{{tabular}}{{{alignment}}}", r"\toprule", header, r"\midrule"]
    else:
        lines += [rf"\begin{{longtable}}{{{alignment}}}",
                  rf"\caption{{{caption(rows, kind)}}}\label{{{label}}} \\",
                  r"\toprule", header, r"\midrule", r"\endfirsthead",
                  r"\caption[]{Performance metrics for the parity-based benchmarks (continued).} \\",
                  r"\toprule", header, r"\midrule", r"\endhead", r"\midrule",
                  rf"\multicolumn{{{len(columns) + 3}}}{{r}}{{Continued on next page.}} \\",
                  r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for row, name in zip(rows, names):
        lines.append("% " + row["instance"].replace("\n", " ").replace("\r", " "))
        cells = [latex_escape(name)]
        success = row["status"] == "ok"
        for key, _ in columns:
            value = row[key]
            if not success and (key not in {field for field, _ in SIZE_COLUMNS} or not value):
                cells.append("---")
            elif key in {field for field, _ in TIMING_COLUMNS}:
                cells.append(format_time(value))
            else:
                number = int(value)
                cells.append(str(number) if number >= 0 else "---")
        cells.append(row["initial_winning"].capitalize() if success else "---")
        cells.append({"ok": "OK", "memory_limit": "ML", "timeout": "TO",
                      "mixed": "Mixed"}.get(row["status"], "Error"))
        lines.append(" & ".join(cells) + r" \\")
    if kind == "ltl":
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    else:
        lines.append(r"\end{longtable}")
    lines += [r"\endgroup", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=ROOT / "benchmark_revealing.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables")
    parser.add_argument("--examples-dir", type=Path, default=ROOT.parents[1] / "examples",
                        help="matching model directory (default: the repository's examples directory)")
    args = parser.parse_args()
    try:
        rows = read_results(args.csv, args.examples_dir)
        tables = {}
        for kind in ("ltl", "parity"):
            selected = [row for row in rows if row["instance"].startswith(kind + "-revealing/")]
            if not selected:
                raise ValueError(f"no {kind} instances in the summary CSV")
            tables[kind] = render_table(selected, kind)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for kind, content in tables.items():
            destination = args.output_dir / f"{kind}_results.tex"
            destination.write_text(content, encoding="utf-8")
            print(f"Written: {destination}")
        print(f"{len(rows)} instances: {dict(Counter(row['status'] for row in rows))}")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
