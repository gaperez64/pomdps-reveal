"""Timing columns and failed runs in the paper's generated tables."""

import csv

import pytest

from results.jair import generate_benchmark_tables as tables


@pytest.fixture
def summary_row():
    return {
        "instance": "ltl-revealing/ltl-guard.pomdp", "status": "ok",
        "initial_winning": "yes", "repetitions": "5", "memory_limit_gb": "24",
        "timeout_seconds": "300", "successful_runs": "5", "timeout_runs": "0",
        "memory_limit_runs": "0", "error_runs": "0",
        "pomdp_states": "4", "pomdp_actions": "4", "pomdp_observations": "13",
        "aut_states": "6", "aut_edges": "48", "bs_states": "24",
        "t_parse": "12.345678", "t_bs": "0.0021", "t_solve": "0.00003",
        "t_total": "45.678912",
    }


def write_summary(path, row):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


@pytest.mark.parametrize("kind", ["ltl", "parity"])
@pytest.mark.parametrize("status", ["ok", "memory_limit"])
def test_table_timings_and_failed_runs(tmp_path, summary_row, kind, status):
    if kind == "parity":
        summary_row["instance"] = "parity-revealing/test.pomdp"
        summary_row["aut_states"] = summary_row["aut_edges"] = ""
    if status != "ok":
        summary_row.update(status=status, successful_runs="0", memory_limit_runs="5",
                           initial_winning="", bs_states="", t_parse="", t_bs="",
                           t_solve="", t_total="")
    path = tmp_path / "summary.csv"
    write_summary(path, summary_row)
    rendered = tables.render_table(tables.read_results(path), kind)
    assert r"$t_{parse}$ & $t_{bs}$ & $t_{solve}$ & $t_{total}$" in rendered
    assert "Times are medians in seconds" in rendered
    if status == "ok":
        assert r" & 12.3457 & 0.0021 & $<0.0001$ & 45.6789 & Yes & OK \\" in rendered
    else:
        assert r" & --- & --- & --- & --- & --- & ML \\" in rendered
    if kind == "parity":
        assert r"\multicolumn{11}{r}{Continued on next page.}" in rendered


@pytest.mark.parametrize("missing", ["t_parse", "t_total"])
def test_new_timing_columns_are_required(tmp_path, summary_row, missing):
    del summary_row[missing]
    path = tmp_path / "summary.csv"
    write_summary(path, summary_row)
    with pytest.raises(ValueError, match=f"missing summary CSV columns: {missing}"):
        tables.read_results(path)
