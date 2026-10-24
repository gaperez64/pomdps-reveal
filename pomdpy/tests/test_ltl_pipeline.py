"""Small LTL examples with independently known winning and losing answers."""

import signal
import subprocess
import sys
from pathlib import Path

import pytest
import spot

import belief_support_algorithm as algorithm
from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.parsers import tlsf as tlsf_parser
from pomdpy.revealing import is_strongly_revealing
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
    pytest.param(0.0, "G F p0;\nG !p1;", "yes", id="safe-retry"),
    pytest.param(0.5, "G F p0;\nG !p1;", "no", id="risky-conjunction"),
    pytest.param(0.5, "G F p0;", "yes", id="liveness-alone"),
    pytest.param(0.5, "G !p1;", "yes", id="safety-alone"),
    pytest.param(0.0, "G F p0;\nG !p0;", "no", id="contradictory-guarantees"),
    pytest.param(0.0, "G F p0 && G(!p1 || !p0);", "yes",
                 id="tlsf-operators"),
    pytest.param(0.0, "G F p1;", "no", id="unreachable-danger"),
])
def test_ltl_benchmark_result(tmp_path, risk, guarantees, winning):
    model_path = write_model(tmp_path, risk)
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
    assert all(row[key] >= 0 for key in benchmark.TIMING_FIELDS)
    assert row["t_total"] >= sum(row[key] for key in ("t_parse", "t_aut", "t_bs", "t_solve"))


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="CLI uses SIGALRM")
def test_ltl_entry_points_check_label_preservation(tmp_path):
    model_path = tmp_path / "labels.pomdp"
    model_path.write_text("""
states: left right
actions: wait
observations: shared left right
atom 0: shared
atom 1: left
atom 2: right
start: uniform
T: wait identity
O: wait
0.5 0.5 0
0.5 0 0.5
""")
    env = pomdp_parser.parse(model_path.read_text())
    assert is_strongly_revealing(env)
    assert not is_strongly_revealing(env, preserve_labels=True)
    write_tlsf(model_path, "G F p0;")
    assert benchmark.run_instance(model_path)["revealing"] == "no"

    result = subprocess.run(
        [sys.executable, str(Path(algorithm.__file__).resolve()),
         "--file", str(model_path), "--ltl_formula", "GFp0"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOT strongly revealing with label-preserving alternatives" in result.stdout
    assert "Results may not be correct when it fails" in result.stdout


def test_missing_tlsf_is_reported_as_an_error(tmp_path):
    model_path = write_model(tmp_path)
    row = benchmark.run_limited_instance(model_path, memory_limit_gb=None,
                                         timeout=5)
    assert row["status"] == f"error: Missing TLSF specification: {model_path.with_suffix('.tlsf')}"
    assert "initial_winning" not in row


@pytest.mark.parametrize("guarantee", ["G (;", r"G \!p1;"])
def test_invalid_ltl_is_reported_as_an_error(tmp_path, guarantee):
    model_path = write_model(tmp_path)
    write_tlsf(model_path, guarantee)
    row = benchmark.run_limited_instance(model_path, memory_limit_gb=None,
                                         timeout=10)
    assert row["status"].startswith("error:"), row
    assert "initial_winning" not in row


def test_tlsf_inline_comment_preserves_losing_guarantee(tmp_path):
    model_path = write_model(tmp_path, risk=0.0)
    path = write_tlsf(model_path, "G F p1; // The danger observation is unreachable.")
    assert spot.are_equivalent(algorithm.parse_tlsf_file(path)["formula"], "GFp1")

    row = benchmark.run_limited_instance(model_path, memory_limit_gb=None, timeout=5)
    assert row["status"] == "ok", row
    assert row["initial_winning"] == "no"


def test_tlsf_truncated_specification_is_an_error(tmp_path):
    model_path = write_model(tmp_path)
    path = write_tlsf(model_path, "false;")
    path.write_text(path.read_text().rstrip()[:-1])
    with pytest.raises(ValueError, match="tlsfinfo failed"):
        algorithm.parse_tlsf_file(path)

    row = benchmark.run_limited_instance(model_path, memory_limit_gb=None, timeout=5)
    assert row["status"].startswith("error:"), row
    assert "initial_winning" not in row


def test_tlsf_sections_in_quoted_metadata_are_ignored(tmp_path):
    model_path = write_model(tmp_path)
    path = write_tlsf(model_path, "false;")
    path.write_text(path.read_text().replace(
        'Reach a goal while avoiding danger',
        r'Example: \"GUARANTEES { true; }\" INPUTS { p99; } https://example.org',
    ))
    assert algorithm.parse_tlsf_file(path) == {"inputs": [0, 1, 2], "formula": "false"}

    row = benchmark.run_limited_instance(model_path, memory_limit_gb=None, timeout=5)
    assert row["status"] == "ok", row
    assert row["initial_winning"] == "no"


@pytest.mark.parametrize("body", ["", " \n ", "// No constraints\n"])
def test_tlsf_explicitly_empty_guarantees_are_allowed(tmp_path, body):
    path = write_tlsf(tmp_path / "model.pomdp", body)
    assert algorithm.parse_tlsf_file(path)["formula"] == "true"


def test_tlsf_parser_preserves_guarantees_and_declared_atoms(tmp_path):
    path = write_tlsf(tmp_path / "model.pomdp", "G F p0;\nG(!p1 || p2);")
    data = algorithm.parse_tlsf_file(path)
    assert data["inputs"] == [0, 1, 2]
    assert spot.are_equivalent(data["formula"], "GFp0 & G(!p1 | p2)")


def test_tlsf_definitions_parameters_and_assumptions(tmp_path):
    path = write_tlsf(tmp_path / "model.pomdp", "G (\n safe && X[delay] p0\n);")
    path.write_text(path.read_text().replace("MAIN {", """
GLOBAL {
  PARAMETERS { delay = 2; }
  DEFINITIONS { safe = !p1; }
}
MAIN {
  ASSUMPTIONS { G F p2; }
"""))
    data = algorithm.parse_tlsf_file(path)
    assert data["inputs"] == [0, 1, 2]
    assert spot.are_equivalent(data["formula"], "GFp2 -> G(!p1 & X X p0)")


def test_tlsf_without_guarantees_preserves_other_constraints(tmp_path):
    path = write_tlsf(tmp_path / "model.pomdp", "!p1;")
    path.write_text(path.read_text().replace("GUARANTEES", "INVARIANTS"))
    assert spot.are_equivalent(algorithm.parse_tlsf_file(path)["formula"], "G !p1")


def test_tlsf_without_inputs(tmp_path):
    path = write_tlsf(tmp_path / "model.pomdp", "true;")
    path.write_text(path.read_text().replace("p0; p1; p2;", ""))
    assert algorithm.parse_tlsf_file(path) == {"inputs": [], "formula": "true"}


@pytest.mark.parametrize("semantics", ["Finite,Moore", "Finite,Mealy"])
def test_tlsf_finite_semantics_are_rejected(tmp_path, semantics):
    path = write_tlsf(tmp_path / "model.pomdp", "G F p0;")
    path.write_text(path.read_text().replace("SEMANTICS: Moore", f"SEMANTICS: {semantics}"))
    with pytest.raises(ValueError, match="Finite-word"):
        algorithm.parse_tlsf_file(path)


@pytest.mark.parametrize("name", ["P0", "p01", "goal"])
def test_tlsf_input_names_must_match_pomdp_convention(tmp_path, name):
    path = write_tlsf(tmp_path / "model.pomdp", "G F p0;")
    path.write_text(path.read_text().replace("p0", name))
    with pytest.raises(ValueError, match=f"unsupported: {name}"):
        algorithm.parse_tlsf_file(path)


def test_tlsf_used_outputs_also_need_pomdp_names(tmp_path):
    path = write_tlsf(tmp_path / "model.pomdp", "G action;")
    with pytest.raises(ValueError, match="unsupported: action"):
        algorithm.parse_tlsf_file(path)


def test_tlsf_missing_tools_have_installation_hint(tmp_path, monkeypatch):
    path = write_tlsf(tmp_path / "model.pomdp", "true;")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="tlsfinfo not found.*install tlsf-tools"):
        algorithm.parse_tlsf_file(path)


def test_tlsf_conversion_timeout(tmp_path, monkeypatch):
    path = write_tlsf(tmp_path / "model.pomdp", "true;")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(tlsf_parser.subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match="tlsfinfo exceeded its 30-second timeout"):
        algorithm.parse_tlsf_file(path)


@pytest.mark.parametrize("formula, winning", [
    ("GFp0 & G!p1", True), ("GFp0 & GFp1", False),
])
def test_algorithm_ltl_result(tmp_path, formula, winning, monkeypatch):
    monkeypatch.setenv("PATH", "")
    env = pomdp_parser.parse(write_model(tmp_path).read_text())
    # Exercise the prebuilt-automaton interface.
    aut = spot.split_edges(spot.translate(
        formula, "parity max even", "deterministic", "complete", "SBAcc", "colored"
    ))
    result = algorithm.belief_support_algorithm(env, formula, parity_automaton=aut)
    mdp = result["belief_support_mdp"]
    assert (mdp.start in result["almost_sure_win_bs_states"]) == winning
    assert result["initial_winning"] == winning
    assert result["parity_automaton"] is aut


def test_ltl_translation_cleanup(tmp_path):
    model_path = write_model(tmp_path)
    # Keep native cleanup failures isolated from the pytest process.
    code = """
import gc
import sys
from pathlib import Path

if sys.platform != "win32":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import pytest
import belief_support_algorithm as algorithm
from pomdpy.parsers import pomdp

env = pomdp.parse(Path(sys.argv[1]).read_text())
# Spot 2.14.1 raises SyntaxError; 2.16 raises spot.parse_error.
# Match their shared diagnostic so unrelated exceptions still fail the test.
with pytest.raises(Exception, match="syntax error"):
    algorithm.belief_support_algorithm(env, "G (", atoms=[0, 1, 2])
gc.collect()

for formula, winning in [("GFp0 & G!p1", True), ("GFp0 & GFp1", False),
                         ("GFp2", True), ("true", True), ("false", False)]:
    result = algorithm.belief_support_algorithm(env, formula, atoms=[0, 1, 2])
    assert (result["belief_support_mdp"].start in
            result["almost_sure_win_bs_states"]) == winning, formula
    del result
    gc.collect()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        cwd=Path(algorithm.__file__).resolve().parent,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    assert result["initial_winning"] == winning
    assert result["almost_sure_win_pomdp_states"] == ({0} if winning else set())
    assert result["parity_automaton"] is None


@pytest.mark.parametrize("mode", ["parity", "ltl", "prebuilt"])
def test_plotting_uses_only_current_files(tmp_path, monkeypatch, mode):
    model_path = write_model(tmp_path)
    source = model_path.read_text()
    if mode == "parity":
        source = source.replace(
            "atom 0: goal danger\natom 1: danger\natom 2: waiting goal danger",
            "prio 0: waiting safe_goal unsafe_goal",
        )
    env = pomdp_parser.parse(source)
    options = {}
    if mode != "parity":
        options["ltl_formula"] = "Fp0"
    if mode == "prebuilt":
        options["parity_automaton"] = spot.split_edges(spot.translate(
            "Fp0", "parity max even", "deterministic", "complete", "SBAcc", "colored"
        ))

    # Old plots must neither be rendered nor reported as generated this time.
    for name in ["pomdp.dot", "automaton.dot", "automaton.png"]:
        (tmp_path / name).write_text("old plot")
    rendered = []

    def render(dot_path, png_path):
        rendered.append(Path(dot_path).name)
        if Path(dot_path).name == "automaton.dot":
            return False
        Path(png_path).write_text("rendered plot")
        return True

    monkeypatch.setattr(algorithm, "HAS_PYGRAPHVIZ", True)
    monkeypatch.setattr(algorithm, "render_dot_to_png", render)
    result = algorithm.belief_support_algorithm(env, plot=True, output_dir=tmp_path, **options)
    files = {Path(path).name for path in result["visualization_files"]}
    expected = {"belief_support_mdp.dot", "belief_support_mdp.png"}
    if mode != "parity":
        expected.add("product_pomdp.dot")
    if mode == "ltl":
        expected.add("automaton.dot")
    assert files == expected
    assert rendered == (["automaton.dot"] if mode == "ltl" else []) + ["belief_support_mdp.dot"]
    assert all(Path(path).is_file() for path in result["visualization_files"])


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="CLI uses SIGALRM")
def test_cli_plot_summary_and_unused_atoms(tmp_path, monkeypatch, capsys):
    model_path = write_model(tmp_path)
    monkeypatch.setattr(algorithm, "HAS_PYGRAPHVIZ", False)
    monkeypatch.setattr(sys, "argv", [
        "belief_support_algorithm", "--file", str(model_path), "--ltl_formula", "Fp0",
        "--plot", "--output_dir", str(tmp_path),
    ])
    assert algorithm.main() == 0
    output = capsys.readouterr().out
    assert "warning" not in output.lower()
    assert "Declaring all POMDP atoms" not in output
    listed = [line.strip()[2:] for line in output.splitlines() if line.startswith("  - ")]
    assert listed == ["automaton.dot", "product_pomdp.dot", "belief_support_mdp.dot"]
    assert all((tmp_path / name).is_file() for name in listed)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="CLI uses SIGALRM")
@pytest.mark.parametrize("guarantees, options", [
    pytest.param("G F p0;\nG !p1;", ["--ltl_formula", "false"],
                 id="conjunction-overrides-formula"),
    pytest.param("", [], id="empty-guarantees"),
    pytest.param("", ["--ltl_formula", "false"],
                 id="empty-guarantees-override-formula"),
])
def test_cli_tlsf_guarantees(tmp_path, guarantees, options):
    import resource

    def disable_core_dumps():
        # Disable core dumps in case native cleanup fails.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    model_path = write_model(tmp_path)
    tlsf_path = write_tlsf(model_path, guarantees)
    result = subprocess.run(
        [sys.executable, str(Path(algorithm.__file__).resolve()),
         "--file", str(model_path), "--verbose",
         "--tlsf_file", str(tlsf_path), *options],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
        preexec_fn=disable_core_dumps,
    )
    assert "COMPUTATION COMPLETE" in result.stdout
    assert "✓ POMDP is strongly revealing with label-preserving alternatives" in result.stdout
    assert "Initial belief is almost-sure winning: yes" in result.stdout
    assert "POMDP states occurring in winning singleton beliefs: [0, 1]" in result.stdout
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="CLI uses SIGALRM")
@pytest.mark.parametrize("start, winning, projected_states", [
    ("start", "yes", "[0, 1, 2]"),
    ("end", "no", "[]"),
])
def test_cli_ltl_winning_states_depend_on_history(tmp_path, start, winning, projected_states):
    # From start, p0 is observed once before entering end forever. The end
    # state then occurs in a winning singleton belief, but a fresh start there loses.
    model_path = tmp_path / "history.pomdp"
    model_path.write_text(f"""
states: start goal end
actions: wait
observations: yes no
atom 0: yes
start include: {start}
T: wait
0 1 0
0 0 1
0 0 1
O: wait
0 1
1 0
0 1
""")
    result = subprocess.run(
        [sys.executable, str(Path(algorithm.__file__).resolve()),
         "--file", str(model_path), "--ltl_formula", "Fp0"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Initial belief is almost-sure winning: {winning}" in result.stdout
    assert f"POMDP states occurring in winning singleton beliefs: {projected_states}" in result.stdout
    assert "For LTL, winning here can depend on past observations" in result.stdout


def test_cli_parity_does_not_require_tlsf_tools(tmp_path, monkeypatch):
    model_path = write_model(tmp_path)
    model_path.write_text(model_path.read_text().replace(
        "atom 0: goal danger\natom 1: danger\natom 2: waiting goal danger",
        "prio 0: waiting safe_goal unsafe_goal",
    ))
    monkeypatch.setenv("PATH", "")
    result = subprocess.run(
        [sys.executable, str(Path(algorithm.__file__).resolve()),
         "--file", str(model_path), "--tlsf_file", "unused.tlsf"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LTL formula/TLSF will be ignored" in result.stdout
    assert "COMPUTATION COMPLETE" in result.stdout
    assert "✓ POMDP is strongly revealing" in result.stdout
    assert "label-preserving alternatives" not in result.stdout
    assert "Initial belief is almost-sure winning: yes" in result.stdout


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
