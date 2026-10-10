"""Resource limits and CSV results for the benchmark runner."""

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
            'initial_winning': 'yes', 'pomdp_states': limit}}

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
    for name in ["1_memory", "2_timeout", "3_crash", "4_error", "5_ok"]:
        (examples / f"{name}.pomdp").touch()

    monkeypatch.setattr(benchmark, "ROOT", str(tmp_path))
    monkeypatch.setattr(benchmark, "TIMEOUT_SECONDS", 1)
    if memory_limit_gb is None:
        monkeypatch.setitem(sys.modules, "resource", None)
        monkeypatch.setattr(sys, "argv", ["benchmark", "--no-memory-limit"])
    else:
        monkeypatch.setattr(sys, "argv", ["benchmark", "--memory-limit-gb", "1"])
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
        completed.append(path.name)
        return row

    monkeypatch.setattr(benchmark, "run_limited_instance", run_and_check_csv)
    benchmark.main()

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["status"] for row in rows] == [
        "memory_limit" if memory_limit_gb is not None else "error: out of memory",
        "timeout", "error: worker exited with code 7",
        "error: invalid instance", "ok",
    ]
    assert [row["initial_winning"] for row in rows] == ["", "", "", "", "yes"]
    expected_limit = "1" if memory_limit_gb is not None else "none"
    assert all(row["memory_limit_gb"] == expected_limit for row in rows)
    assert all(row["timeout_seconds"] == "1" for row in rows)
    # Limit observed inside the worker (blank when no limit was requested).
    assert rows[-1]["pomdp_states"] == ("1" if memory_limit_gb is not None else "")
    pid = int((examples / "2_timeout.pid").read_text())
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


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
