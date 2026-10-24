"""Resource limits, repetitions, and CSV results for the benchmark runner."""

import csv
import json
import os
import sys
from types import SimpleNamespace

import pytest

from scripts import run_benchmark_revealing as benchmark


@pytest.fixture
def synthetic_worker(tmp_path, monkeypatch):
    """Exercise real child processes and OS limits without large benchmarks."""
    worker = tmp_path / "worker.py"
    worker.write_text(f"""
import os
import sys
import time

if '--no-memory-limit' in sys.argv:
    # Simulate a platform where importing resource is not possible.
    sys.modules['resource'] = None

sys.path.insert(0, {benchmark.ROOT!r})
from scripts import run_benchmark_revealing as benchmark

def run_instance(path):
    if path.stem.endswith('memory'):
        if '--no-memory-limit' in sys.argv:
            raise MemoryError('out of memory')
        # Refused by the 1 GiB limit before allocating any physical memory.
        bytearray(2 * 1024 ** 3)
    elif path.stem.endswith('timeout'):
        path.with_suffix('.pid').write_text(str(os.getpid()))
        time.sleep(60)
    elif path.stem.endswith('crash'):
        os._exit(7)
    elif path.stem.endswith('error'):
        raise ValueError('invalid instance')
    limit = None
    if '--no-memory-limit' not in sys.argv:
        import resource
        limit = resource.getrlimit(resource.RLIMIT_AS)[0] // 1024 ** 3
        assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
    return {{'instance': path.name, 'status': 'ok',
            'initial_winning': 'yes', 'pomdp_states': limit,
            't_parse': 0.03, 't_aut': 0.0, 't_bs': 0.01, 't_solve': 0.02}}

benchmark.run_instance = run_instance
sys.exit(benchmark.main())
""")
    monkeypatch.setattr(benchmark, "__file__", str(worker))


@pytest.mark.parametrize("memory_limit_gb", [1, None])
def test_batch_continues_after_failures_and_flushes_csv(tmp_path, monkeypatch,
                                                       synthetic_worker, memory_limit_gb):
    if memory_limit_gb is not None and sys.platform != "linux":
        pytest.skip("OS memory-limit enforcement is tested on Linux")
    examples = tmp_path / "examples" / "parity-revealing"
    examples.mkdir(parents=True)
    for name in ["1_memory", "2_timeout", "3_crash", "4_error", "5_ok", "variant/5_ok"]:
        path = examples / f"{name}.pomdp"
        path.parent.mkdir(exist_ok=True)
        path.touch()

    monkeypatch.setattr(benchmark, "ROOT", str(tmp_path))
    monkeypatch.setattr(benchmark, "TIMEOUT_SECONDS", 1)
    if memory_limit_gb is None:
        monkeypatch.setitem(sys.modules, "resource", None)
        monkeypatch.setattr(sys, "argv", ["benchmark", "--no-memory-limit", "--repetitions", "1"])
    else:
        monkeypatch.setattr(sys, "argv", ["benchmark", "--memory-limit-gb", "1", "--repetitions", "1"])
    csv_path = examples.parent / "benchmark_revealing.csv"
    run_limited_instance = benchmark.run_limited_instance
    completed = []

    def run_and_check_csv(path, memory_limit_gb, timeout):
        # Read while the writer is still open: earlier rows must be flushed.
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert "initial_winning" in reader.fieldnames
            assert [row["instance"] for row in reader] == completed
        row = run_limited_instance(path, memory_limit_gb, timeout)
        completed.append(path.relative_to(examples.parent).as_posix())
        return row

    monkeypatch.setattr(benchmark, "run_limited_instance", run_and_check_csv)
    benchmark.main()

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["status"] for row in rows] == [
        "memory_limit" if memory_limit_gb is not None else "error: out of memory",
        "timeout", "error: worker exited with code 7",
        "error: invalid instance", "ok", "ok",
    ]
    assert [row["initial_winning"] for row in rows] == ["", "", "", "", "yes", "yes"]
    assert [row["instance"] for row in rows] == completed
    assert [row["instance"] for row in rows[-2:]] == [
        "parity-revealing/5_ok.pomdp", "parity-revealing/variant/5_ok.pomdp",
    ]
    expected_limit = "1" if memory_limit_gb is not None else "none"
    assert all(row["memory_limit_gb"] == expected_limit for row in rows)
    assert all(row["timeout_seconds"] == "1" for row in rows)
    assert all(row["repetitions"] == "1" for row in rows)
    assert [row["successful_runs"] for row in rows] == ["0", "0", "0", "0", "1", "1"]
    for row in rows[-2:]:
        for field in benchmark.TIMING_FIELDS:
            assert row[f"{field}_stddev"] == ""
            for statistic in ("mean", "min", "max"):
                assert float(row[f"{field}_{statistic}"]) == float(row[field])
    assert all(value == "" for row in rows[:4] for key, value in row.items() if key.startswith("t_"))
    with (examples.parent / "benchmark_revealing_runs.csv").open() as f:
        raw = list(csv.DictReader(f))
    assert all(float(row["t_total"]) > 0 for row in raw)
    assert float(raw[1]["t_total"]) >= 1  # Includes waiting for the timeout.
    assert all(row["t_parse"] == "" for row in raw[:4])
    assert all(float(row["t_parse"]) == 0.03 for row in raw[-2:])
    # Limit observed inside the worker (blank when no limit was requested).
    assert rows[-1]["pomdp_states"] == ("1" if memory_limit_gb is not None else "")
    pid = int((examples / "2_timeout.pid").read_text())
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("repetitions", [None, 4])
def test_repeated_runs_keep_raw_results_and_report_statistics(tmp_path, monkeypatch, repetitions):
    examples = tmp_path / "examples"
    models = examples / "parity-revealing"
    models.mkdir(parents=True)
    for name in ("a_ok", "b_mixed"):
        (models / f"{name}.pomdp").touch()
    monkeypatch.setattr(benchmark, "ROOT", str(tmp_path))
    argv = ["benchmark", "--no-memory-limit"]
    if repetitions is not None:
        argv += ["--repetitions", str(repetitions)]
    monkeypatch.setattr(sys, "argv", argv)
    count = 5 if repetitions is None else repetitions
    calls = []
    durations = [9.0, 1.0, 3.0, 2.0, 5.0]
    factors = {"t_parse": 4, "t_aut": 1, "t_bs": 2, "t_solve": 3, "t_total": 11}
    failures = ["ok", "timeout", "memory_limit", "error: invalid instance", "ok"]

    def run_instance(path, memory_limit_gb, timeout):
        assert memory_limit_gb is None
        assert timeout == 300
        # Both CSVs must retain completed work even if a later run is interrupted.
        with (examples / "benchmark_revealing_runs.csv").open() as f:
            raw = list(csv.DictReader(f))
        assert [(row["instance"], int(row["run"])) for row in raw] == calls
        with (examples / "benchmark_revealing.csv").open() as f:
            assert len(list(csv.DictReader(f))) == len(calls) // count
        run = len(calls) % count + 1
        calls.append((path.relative_to(examples).as_posix(), run))
        status = "ok" if path.stem == "a_ok" else failures[run - 1]
        if status != "ok":
            return {"status": status, "t_total": factors["t_total"] * durations[run - 1]}
        return {"status": "ok", "initial_winning": "yes", "bs_states": 7,
                **{field: factor * durations[run - 1] for field, factor in factors.items()}}

    monkeypatch.setattr(benchmark, "run_limited_instance", run_instance)
    benchmark.main()

    with (examples / "benchmark_revealing.csv").open() as f:
        success, mixed = list(csv.DictReader(f))
    expected = 3.0 if count == 5 else 2.5
    assert success["status"] == "ok"
    assert success["initial_winning"] == "yes"
    assert success["bs_states"] == "7"
    expected_mean = 4.0 if count == 5 else 3.75
    expected_stddev = 10 ** 0.5 if count == 5 else (155 / 12) ** 0.5
    for field, factor in factors.items():
        assert float(success[field]) == factor * expected
        assert float(success[f"{field}_mean"]) == factor * expected_mean
        assert float(success[f"{field}_stddev"]) == pytest.approx(factor * expected_stddev)
        assert float(success[f"{field}_min"]) == factor * 1.0
        assert float(success[f"{field}_max"]) == factor * 9.0
    assert success["repetitions"] == success["successful_runs"] == str(count)
    assert mixed["status"] == "mixed"
    assert mixed["successful_runs"] == str(count - 3)
    assert mixed["timeout_runs"] == mixed["memory_limit_runs"] == mixed["error_runs"] == "1"
    assert mixed["initial_winning"] == ""
    assert all(value == "" for key, value in mixed.items() if key.startswith("t_"))
    with (examples / "benchmark_revealing_runs.csv").open() as f:
        raw = list(csv.DictReader(f))
    assert len(raw) == 2 * count
    assert [int(row["run"]) for row in raw] == list(range(1, count + 1)) * 2
    assert [row["status"] for row in raw[count:]] == failures[:count]
    assert [float(row["t_aut"]) for row in raw[:count]] == durations[:count]
    assert [float(row["t_total"]) for row in raw] == [
        factors["t_total"] * duration for duration in durations[:count]
    ] * 2
    assert all(row["memory_limit_gb"] == "none" and row["timeout_seconds"] == "300" for row in raw)


@pytest.mark.parametrize("field, value", [("initial_winning", "no"), ("bs_states", 8)])
def test_inconsistent_repetitions_are_not_summarized_as_success(field, value):
    row = {"instance": "test.pomdp", "memory_limit_gb": 24, "timeout_seconds": 300,
           "status": "ok", "initial_winning": "yes", "bs_states": 7,
           "t_parse": 0.03, "t_aut": 0.0, "t_bs": 0.01, "t_solve": 0.02, "t_total": 0.1}
    summary = benchmark.summarize_runs([row, {**row, field: value}])
    assert summary["status"] == "error: inconsistent results across repetitions"
    assert not any(key.startswith("t_") for key in summary)
    assert "initial_winning" not in summary


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_invalid_repetitions(value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--repetitions", value])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2


@pytest.mark.skipif(sys.platform != "linux", reason="OS memory-limit enforcement is tested on Linux")
def test_default_memory_limit_and_parent_unchanged(tmp_path, synthetic_worker):
    import resource

    parent_limit = resource.getrlimit(resource.RLIMIT_AS)
    row = benchmark.run_limited_instance(tmp_path / "ok.pomdp")
    assert row["status"] == "ok"
    assert row["pomdp_states"] == 16
    assert resource.getrlimit(resource.RLIMIT_AS) == parent_limit


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid", str(2 ** 64)])
def test_invalid_memory_limit(value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--memory-limit-gb", value])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2


def test_memory_options_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--memory-limit-gb", "8", "--no-memory-limit"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("resource_module", [None, SimpleNamespace()])
def test_unavailable_limits_require_explicit_opt_out(resource_module, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "resource", resource_module)
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert "--no-memory-limit" in capsys.readouterr().err


def test_rejected_limits_do_not_run_the_solver(tmp_path, monkeypatch):
    def reject_limit(*args):
        raise OSError("unsupported limit")

    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RLIMIT_AS=1, RLIM_INFINITY=-1,
        getrlimit=lambda _: (-1, -1), setrlimit=reject_limit,
    ))
    monkeypatch.setattr(benchmark, "run_instance", lambda _: pytest.fail("solver must not run"))
    result_file = tmp_path / "result.json"
    assert benchmark.run_worker(tmp_path / "instance.pomdp", result_file, 16) == 0
    row = json.loads(result_file.read_text())
    assert row["status"].startswith("error: cannot set resource limits")
    assert "--no-memory-limit" in row["status"]


@pytest.mark.parametrize("memory_limit_gb", [16, None])
@pytest.mark.parametrize("priority, winning", [(0, "yes"), (1, "no")])
def test_isolated_solver_result(tmp_path, priority, winning, memory_limit_gb):
    if memory_limit_gb is not None and sys.platform != "linux":
        pytest.skip("OS memory-limit enforcement is tested on Linux")
    pomdp_file = tmp_path / "one_state.pomdp"
    pomdp_file.write_text(f"""
states: s
actions: a
observations: o
prio {priority}: s
start include: s
T: a identity
O: a uniform
""")
    row = benchmark.run_limited_instance(pomdp_file, memory_limit_gb)
    assert row["status"] == "ok"
    assert row["initial_winning"] == winning
    assert row["pomdp_states"] == 1
    assert row["bs_states"] == 1
    assert row["revealing"] == "yes"
    assert all(row[field] >= 0 for field in benchmark.TIMING_FIELDS)
    assert row["t_total"] >= sum(row[field] for field in ("t_parse", "t_aut", "t_bs", "t_solve"))


def test_parse_timing_includes_file_loading(tmp_path, monkeypatch):
    from pomdpy.parsers import pomdp as pomdp_parser

    pomdp_file = tmp_path / "one_state.pomdp"
    pomdp_file.write_text("""
states: s
actions: a
observations: o
prio 0: s
start include: s
T: a identity
O: a uniform
""")
    elapsed = 0.0
    parse = pomdp_parser.parse

    def open_model(*args, **kwargs):
        nonlocal elapsed
        elapsed += 2.0
        return open(*args, **kwargs)

    def parse_model(content):
        nonlocal elapsed
        elapsed += 3.0
        return parse(content)

    monkeypatch.setattr(benchmark, "open", open_model, raising=False)
    monkeypatch.setattr(pomdp_parser, "parse", parse_model)
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: elapsed))
    row = benchmark.run_instance(pomdp_file)
    assert row["initial_winning"] == "yes"
    assert row["t_parse"] == 5.0
    assert row["t_aut"] == row["t_bs"] == row["t_solve"] == 0.0
