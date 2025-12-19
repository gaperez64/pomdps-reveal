#!/usr/bin/env python3
"""
Test the strongly revealing POMDP implementation (pytest).
"""

from pathlib import Path
import os
import pytest

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.revealing import is_strongly_revealing, make_strongly_revealing

def _find_examples_dir():
    """
    Find the examples directory by:
    - POMDPY_EXAMPLES_DIR env var, or
    - walking up parent directories to locate examples.
    """
    env = os.environ.get("POMDPY_EXAMPLES_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "examples"
        if candidate.exists():
            return candidate

    pytest.skip("examples directory not found; set POMDPY_EXAMPLES_DIR or ensure repository layout")

def _read_example_file(name: str) -> Path:
    examples_dir = _find_examples_dir()
    path = examples_dir / name
    if not path.exists():
        pytest.skip(f"Example file not found: {path}")
    return path

def test_original_is_strongly_revealing():
    """The ltl-revealing tiger (repeating) example should be strongly revealing."""
    filepath = _read_example_file("ltl-revealing/ltl-tiger-repeating.pomdp")
    content = filepath.read_text()
    pomdp = pomdp_parser.parse(content)
    assert is_strongly_revealing(pomdp) is True

def test_transformation_makes_strongly_revealing():
    """Transform a POMDP to be strongly revealing."""
    filepath = _read_example_file("ltl/ltl-guard.pomdp")
    pomdp_str = open(filepath).read()
    pomdp = pomdp_parser.parse(pomdp_str)
    
    assert not is_strongly_revealing(pomdp)
    
    revealing_pomdp = make_strongly_revealing(pomdp)
    assert is_strongly_revealing(revealing_pomdp)
