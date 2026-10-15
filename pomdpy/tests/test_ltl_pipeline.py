"""Small LTL examples with independently known winning and losing answers."""

import signal
import subprocess
import sys
from pathlib import Path

import pytest
import spot

import belief_support_algorithm as algorithm
from pomdpy.parsers import pomdp as pomdp_parser
from scripts import run_benchmark_revealing as benchmark


def write_model(tmp_path, risk=0.0):
    # Waiting forever is safe but never reaches a goal. Trying repeatedly
    # reaches a goal almost surely, but may enter the unsafe absorbing goal.
    path = tmp_path / "goals.pomdp"
    path.write_text(f"""
states: waiting safe_goal unsafe_goal
actions: wait try
observations: waiting goal danger
atom 0: goal danger
atom 1: danger
atom 2: waiting goal danger
start include: waiting
T: wait identity
T: try
0.5 {0.5 * (1 - risk)} {0.5 * risk}
0 1 0
0 0 1
O: *
1 0 0
0 1 0
0 0 1
""")
    return path


def write_tlsf(model_path, guarantees):
    path = model_path.with_suffix(".tlsf")
    path.write_text(f"""INFO {{
  TITLE: "Goals"
  DESCRIPTION: "Reach a goal while avoiding danger"
  SEMANTICS: Moore
  TARGET: Moore
}}
MAIN {{
  INPUTS {{ p0; p1; p2; }}
  OUTPUTS {{ action; }}
  GUARANTEES {{
    // All guarantees must hold.

    {guarantees}
  }}
}}
""")
    return path


@pytest.mark.parametrize("risk, guarantees, winning", [
    pytest.param(0.0, "GFp0;\nG!p1;", "yes", id="safe-retry"),
    pytest.param(0.5, "GFp0;\nG!p1;", "no", id="risky-conjunction"),
    pytest.param(0.5, "GFp0;", "yes", id="liveness-alone"),
    pytest.param(0.5, "G!p1;", "yes", id="safety-alone"),
    pytest.param(0.0, "GFp0;\nG!p0;", "no", id="contradictory-guarantees"),
    pytest.param(0.0, r"GFp0 && G(\!p1 || \!p0);", "yes",
                 id="tlsf-operators"),
    pytest.param(0.0, "GFp1;", "no", id="unreachable-danger"),
    pytest.param(0.5, None, "yes", id="missing-tlsf-default-liveness"),
])
def test_ltl_benchmark_result(tmp_path, risk, guarantees, winning):
    model_path = write_model(tmp_path, risk)
    if guarantees is not None:
        write_tlsf(model_path, guarantees)

    row = benchmark.run_limited_instance(model_path, memory_limit_gb=None,
                                         timeout=5)
    assert row["status"] == "ok", row
    assert row["initial_winning"] == winning
    assert row["pomdp_type"] == "atomic"
    assert row["pomdp_states"] == 3
    assert row["revealing"] == "yes"
    # Do not depend on Spot's particular automaton representation or timings.
    assert row["aut_states"] > 0
    assert row["aut_edges"] > 0
    assert row["bs_states"] > 0
    assert all(row[key] >= 0 for key in ["t_aut", "t_bs", "t_solve"])


def test_invalid_ltl_is_reported_as_an_error(tmp_path):
    model_path = write_model(tmp_path)
    write_tlsf(model_path, "G (;")
    row = benchmark.run_limited_instance(model_path, memory_limit_gb=None,
                                         timeout=10)
    assert row["status"].startswith("error:"), row
    assert "initial_winning" not in row


def test_tlsf_parser_preserves_guarantees_and_declared_atoms(tmp_path):
    path = write_tlsf(tmp_path / "model.pomdp", "GFp0;\n" + r"G(\!p1 || p2);")
    assert algorithm.parse_tlsf_file(path) == {
        "inputs": [0, 1, 2],
        "formulas": ["GFp0", "G(!p1 | p2)"],
    }


@pytest.mark.parametrize("formula, winning", [
    ("GFp0 & G!p1", True), ("GFp0 & GFp1", False),
])
def test_algorithm_ltl_result(tmp_path, formula, winning):
    env = pomdp_parser.parse(write_model(tmp_path).read_text())
    # Exercise the prebuilt-automaton interface. The default translation path
    # is tested in a subprocess below because its cleanup currently aborts.
    aut = spot.split_edges(spot.translate(
        formula, "parity max even", "deterministic", "complete", "SBAcc", "colored"
    ))
    result = algorithm.belief_support_algorithm(env, formula, parity_automaton=aut)
    mdp = result["belief_support_mdp"]
    assert (mdp.start in result["almost_sure_win_bs_states"]) == winning
    assert result["parity_automaton"] is aut


@pytest.mark.parametrize("priority, winning", [
    (0, True),
    (1, False),
])
def test_algorithm_parity_result(priority, winning):
    env = pomdp_parser.parse(f"""
states: s
actions: a
observations: o
prio {priority}: s
start include: s
T: a identity
O: a uniform
""")
    result = algorithm.belief_support_algorithm(env)
    mdp = result["belief_support_mdp"]
    assert (mdp.start in result["almost_sure_win_bs_states"]) == winning
    assert result["almost_sure_win_pomdp_states"] == ({0} if winning else set())
    assert result["parity_automaton"] is None


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="CLI uses SIGALRM")
@pytest.mark.xfail(
    raises=ChildProcessError, strict=True,
    reason="CLI aborts during Spot dictionary cleanup: propositions remain registered",
)
def test_cli_tlsf_takes_precedence_over_formula(tmp_path):
    import resource

    def disable_core_dumps():
        # The known native crash should not leave a core file on each test run.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    model_path = write_model(tmp_path)
    tlsf_path = write_tlsf(model_path, "GFp0;\nG!p1;")
    result = subprocess.run(
        [sys.executable, str(Path(algorithm.__file__).resolve()),
         "--file", str(model_path), "--verbose",
         "--tlsf_file", str(tlsf_path), "--ltl_formula", "false"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
        preexec_fn=disable_core_dumps,
    )
    assert "COMPUTATION COMPLETE" in result.stdout
    assert "Almost-sure winning POMDP states: [0, 1]" in result.stdout
    if (result.returncode == -signal.SIGABRT
            and "some maps are not empty" in result.stderr):
        raise ChildProcessError("Spot dictionary cleanup aborted the CLI")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("options, message", [
    ([], "requires LTL formula"),
    (["--ltl_formula", "GFp0", "--atoms", "p0"], "Invalid atoms specification"),
    (["--tlsf_file", "missing.tlsf"], "not found"),
])
def test_cli_rejects_invalid_specification(tmp_path, options, message):
    model_path = write_model(tmp_path)
    result = subprocess.run(
        [sys.executable, str(Path(algorithm.__file__).resolve()),
         "--file", str(model_path), *options],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    assert "COMPUTATION COMPLETE" not in result.stdout
