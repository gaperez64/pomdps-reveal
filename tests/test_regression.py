import pytest
import os
import subprocess
from pathlib import Path

from pomdpy.parsers.pomdp import parse
# POMDP class is not directly used in tests after refactoring, but parse() returns an instance of it.
# from pomdpy.pomdp import POMDP 

# Fixtures

@pytest.fixture(scope="session")
def repo_root():
    """Calculates and returns the absolute path to the repository root."""
    # Assumes tests/ is directly under the repo root
    return Path(__file__).parent.parent.resolve()

@pytest.fixture
def pomdp_file_path(repo_root, example_filename):
    """Constructs the full path to an example POMDP file."""
    return repo_root / "examples" / example_filename

@pytest.fixture
def golden_file_content(repo_root, golden_filepath_relative):
    """Reads and returns the stripped content of a golden file."""
    full_path = repo_root / golden_filepath_relative
    with open(full_path, 'r') as f:
        # Strip individual lines and then the whole content
        lines = [line.strip() for line in f.readlines()]
        return "\n".join(lines).strip()

# Parser Tests

PARSER_TEST_CASES = [
    ("tiger.pomdp", 
     ['tiger-left', 'tiger-right'], 
     ['listen', 'open-left', 'open-right'], 
     ['tiger-left', 'tiger-right']),
    ("kaspers.pomdp", 
     ['s0', 's1', 's2', 's3', 's4'], 
     ['a', 'b'], 
     ['init', 'o1', 'o2', 'o3', 'o5']),
    ("pierres.pomdp", 
     ['s0', 's1', 's2', 's3', 's4', 's5', 's6'], 
     ['a', 'b'], 
     ['init', 'o1', 'o2', 'dead', 'done']),
    ("revealing-tiger.pomdp", 
     ['tiger-left', 'tiger-right', 'dead', 'done'], 
     ['listen', 'open-left', 'open-right'], 
     ['maybe-left', 'maybe-right', 'defo-left', 'defo-right', 'dead-obs', 'done-obs']),
]

@pytest.mark.parametrize(
    "example_filename, expected_states, expected_actions, expected_observations",
    PARSER_TEST_CASES
)
def test_parse_pomdp_files(repo_root, example_filename, expected_states, expected_actions, expected_observations):
    """Tests parsing of various POMDP files."""
    # pomdp_file_path fixture is implicitly used here by pytest due to matching name
    file_path = repo_root / "examples" / example_filename
    pomdp_model = parse(str(file_path)) # parse function expects a string path

    assert pomdp_model is not None
    assert pomdp_model.states == expected_states
    assert pomdp_model.actions == expected_actions
    assert pomdp_model.observations == expected_observations

# Aswin.py Script Tests

ASWIN_TEST_CASES = [
    ("tiger.pomdp", ['--cobuchi', 'tiger-left'], 
     "tests/expected_outputs/aswin/tiger.pomdp.cobuchi_tiger-left.txt"),
    ("kaspers.pomdp", ['--cobuchi', 's0'], 
     "tests/expected_outputs/aswin/kaspers.pomdp.cobuchi_s0.txt"),
    ("pierres.pomdp", ['--cobuchi', 's5'], 
     "tests/expected_outputs/aswin/pierres.pomdp.cobuchi_s5.txt"),
    ("pierres.pomdp", ['--cobuchi', 's5', '--buchi', 's6'], 
     "tests/expected_outputs/aswin/pierres.pomdp.cobuchi_s5_buchi_s6.txt"),
    ("revealing-tiger.pomdp", ['--cobuchi', 'dead', '--buchi', 'done'], 
     "tests/expected_outputs/aswin/revealing-tiger.pomdp.cobuchi_dead_buchi_done.txt"),
    ("revealing-tiger.pomdp", ['--cobuchi', 'tiger-left', 'tiger-right', '--buchi', 'done'], 
     "tests/expected_outputs/aswin/revealing-tiger.pomdp.cobuchi_tiger-left_tiger-right_buchi_done.txt"),
]

@pytest.mark.parametrize(
    "pomdp_filename, aswin_args, golden_filepath_relative",
    ASWIN_TEST_CASES
)
def test_aswin_script(repo_root, pomdp_filename, aswin_args, golden_filepath_relative, golden_file_content):
    """Tests aswin.py script against golden files."""
    aswin_script_path = str(repo_root / "aswin.py")
    full_pomdp_path = str(repo_root / "examples" / pomdp_filename)

    command = ['python', aswin_script_path, full_pomdp_path] + aswin_args
    
    process = subprocess.run(command, capture_output=True, text=True, check=True, cwd=str(repo_root))
    actual_output = process.stdout.strip()
    
    # golden_file_content fixture is used here
    expected_output = golden_file_content 
    assert actual_output == expected_output

# Sim.py Script Tests

SIM_TEST_CASES = [
    ("tiger.pomdp", "listen\nopen-left\n", 
     "tests/expected_outputs/sim/tiger.pomdp.listen_open-left.txt"),
    ("kaspers.pomdp", "a\nb\n", 
     "tests/expected_outputs/sim/kaspers.pomdp.a_b.txt"),
    ("pierres.pomdp", "a\nb\n", 
     "tests/expected_outputs/sim/pierres.pomdp.a_b.txt"),
    ("revealing-tiger.pomdp", "listen\nopen-right\n", 
     "tests/expected_outputs/sim/revealing-tiger.pomdp.listen_open-right.txt"),
]

@pytest.mark.parametrize(
    "pomdp_filename, actions_string, golden_filepath_relative",
    SIM_TEST_CASES
)
def test_sim_script(repo_root, pomdp_filename, actions_string, golden_filepath_relative, golden_file_content):
    """Tests sim.py script against golden files with filtered output."""
    sim_script_path = str(repo_root / "sim.py")
    full_pomdp_path = str(repo_root / "examples" / pomdp_filename)

    command = ['python', sim_script_path, full_pomdp_path, '-s']
    
    process = subprocess.run(command, input=actions_string, capture_output=True, text=True, cwd=str(repo_root))
    # Do not use check=True as EOFError is expected
    
    raw_stdout = process.stdout
    
    filtered_lines = []
    for line in raw_stdout.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("Enter an action:") or \
           line_stripped.startswith("New observation ="):
            filtered_lines.append(line_stripped)
    actual_filtered_output = "\n".join(filtered_lines).strip()
    
    # golden_file_content fixture is used here
    expected_output = golden_file_content
    assert actual_filtered_output == expected_output

# To run these tests, navigate to the repository root in the terminal and run:
# pytest tests/test_regression.py
# Ensure pytest is installed: pip install pytest
# Ensure all dependencies from previous steps are installed.
