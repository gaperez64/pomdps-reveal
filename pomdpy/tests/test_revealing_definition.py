"""Regression cases for Definition 5.1 and the signal-labelling condition (2)."""

from copy import deepcopy
import subprocess
import sys
from pathlib import Path

import pytest

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.pomdp import POMDP, AtomicPropPOMDP, ParityPOMDP
from pomdpy.revealing import is_strongly_revealing, make_strongly_revealing


ROOT = Path(__file__).resolve().parents[2]


def make_model(model_type=POMDP, observations=("common",)):
    model = model_type()
    model.setStates(["left", "right"])
    model.setActions(["act"])
    model.setObs(list(observations))
    model.start = {0: 0.3, 1: 0.7}
    model.addIdentityTrans()
    model.addUniformObs()
    if isinstance(model, AtomicPropPOMDP):
        model.addAtom(0, [observations[0]])
    if isinstance(model, ParityPOMDP):
        model.addPriority(1, ["left"])
        model.addPriority(2, ["right"])
    model.computeTrans()
    return model


def labelled_model():
    model = make_model(AtomicPropPOMDP, ("common", "left-only", "right-only"))
    model.O = {0: {0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 2: 0.5}}}
    model.addAtom(1, ["left-only"])
    model.transitions = {}
    model.computeTrans()
    return model


def label(model, observation):
    return frozenset(atom for atom, observations in getattr(model, "atoms", {}).items()
                     if observation in observations)


def labelled_marginal(model, outcomes):
    result = {}
    for (destination, observation), probability in outcomes.items():
        if probability > 0:
            key = (destination, label(model, observation))
            result[key] = result.get(key, 0.0) + probability
    return result


def test_signal_must_reveal_across_sources_including_unreachable_states():
    model = make_model()
    model.start = {0: 1.0}
    # Each source has a deterministic successor, but the common signal is ambiguous.
    assert not is_strongly_revealing(model)


def test_revealing_signal_is_specific_to_the_action():
    model = make_model()
    model.setActions(["go-left", "go-right"])
    model.transitions = {state: {0: {(0, 0): 1.0}, 1: {(1, 0): 1.0}}
                         for state in range(2)}
    assert is_strongly_revealing(model)


def test_alternative_must_be_available_from_the_same_source():
    model = make_model(observations=("common", "left-only", "right-only"))
    model.transitions = {
        0: {0: {(0, 0): 1.0}},
        1: {0: {(0, 1): 0.5, (1, 0): 0.25, (1, 2): 0.25}},
    }
    assert not is_strongly_revealing(model)


def test_zero_probability_outcomes_do_not_prevent_revelation():
    model = labelled_model()
    model.transitions[1][0][(1, 1)] = 0.0
    assert is_strongly_revealing(model)


def test_labelling_condition_is_stronger_than_strong_revelation():
    model = labelled_model()
    assert is_strongly_revealing(model)
    assert not is_strongly_revealing(model, preserve_labels=True)


@pytest.mark.parametrize("model_type", [POMDP, AtomicPropPOMDP, ParityPOMDP])
def test_transformation_preserves_dynamics_and_metadata(model_type):
    model = make_model(model_type, ("common", "another"))
    transformed = make_strongly_revealing(model)
    assert isinstance(transformed, model_type)
    assert transformed.start == model.start
    assert transformed.T == model.T
    assert is_strongly_revealing(transformed, preserve_labels=True)
    for state, actions in model.transitions.items():
        for action, outcomes in actions.items():
            actual = transformed.transitions[state][action]
            assert sum(actual.values()) == pytest.approx(1.0)
            assert labelled_marginal(transformed, actual) == pytest.approx(
                labelled_marginal(model, outcomes)
            )
    if isinstance(model, ParityPOMDP):
        assert transformed.prio == model.prio
        assert transformed.prioinv == model.prioinv


@pytest.mark.parametrize("model_type", [POMDP, AtomicPropPOMDP, ParityPOMDP])
def test_transformation_keeps_factorization_consistent(model_type):
    model = make_model(model_type, ("common", "another"))
    transformed = make_strongly_revealing(model)
    recomputed = deepcopy(transformed)
    recomputed.transitions = {}
    recomputed.computeTrans()
    for state, actions in transformed.transitions.items():
        for action, outcomes in actions.items():
            for outcome, probability in recomputed.transitions[state][action].items():
                assert outcomes.get(outcome, 0.0) == pytest.approx(probability)


@pytest.mark.parametrize("model_type", [POMDP, AtomicPropPOMDP, ParityPOMDP])
def test_transformation_is_idempotent(model_type):
    model = make_model(model_type, ("common", "another"))
    transformed = make_strongly_revealing(model)
    assert make_strongly_revealing(transformed).__dict__ == transformed.__dict__


@pytest.mark.parametrize("model_type", [POMDP, AtomicPropPOMDP, ParityPOMDP])
def test_transformation_returns_an_independent_copy(model_type):
    model = make_model(model_type, ("common", "another"))
    original = deepcopy(model.__dict__)
    transformed = make_strongly_revealing(model)
    assert model.__dict__ == original
    # Mutating the copy must not affect the input's nested dictionaries.
    transformed.transitions[0][0].clear()
    transformed.transitions[0].clear()
    transformed.transitions.clear()
    transformed.T[0][0].clear()
    assert model.__dict__ == original


@pytest.mark.parametrize("epsilon", [0.05, 0.2, 0.7])
def test_labelled_alternatives_split_mass_without_changing_existing_revelations(epsilon):
    model = labelled_model()
    transformed = make_strongly_revealing(model, epsilon=epsilon)
    assert is_strongly_revealing(transformed, preserve_labels=True)
    for state in range(2):
        row = transformed.transitions[state][0]
        assert row[(state, 0)] == pytest.approx(0.5 * (1 - epsilon))
        assert row[(state, state + 1)] == 0.5
        copies = [(obs, prob) for (dest, obs), prob in row.items()
                  if dest == state and obs >= len(model.obs) and prob > 0]
        assert len(copies) == 1
        observation, probability = copies[0]
        assert probability == pytest.approx(0.5 * epsilon)
        assert label(transformed, observation) == label(model, 0)


def test_generated_names_do_not_reuse_existing_signals():
    model = make_model(observations=("common", "reveal_left"))
    transformed = make_strongly_revealing(model)
    assert len(set(transformed.obs)) == len(transformed.obs)
    assert transformed.obs[:len(model.obs)] == model.obs
    assert "reveal_left_1" in transformed.obs
    assert is_strongly_revealing(transformed)


@pytest.mark.parametrize("factored", [True, False])
def test_revealing_copies_are_shared_only_within_a_destination_and_label(factored):
    model = make_model(AtomicPropPOMDP, ("first", "second", "third"))
    model.addAtom(0, ["first", "second"])
    if not factored:
        model.T = {}
        model.O = {}
    transformed = make_strongly_revealing(model)
    assert len(transformed.obs) == len(model.obs) + 4
    assert is_strongly_revealing(transformed, preserve_labels=True)
    for state in range(2):
        copies = {label(transformed, obs): p
                  for (dest, obs), p in transformed.transitions[state][0].items()
                  if dest == state and obs >= len(model.obs) and p > 0}
        assert copies == pytest.approx({
            frozenset({0}): 0.05 * 2 / 3,
            frozenset(): 0.05 / 3,
        })
    if factored:
        loaded = pomdp_parser.parse(transformed.to_pomdp_string())
        assert is_strongly_revealing(loaded, preserve_labels=True)
        for state in range(2):
            actual = {obs: p for obs, p in loaded.O[0][state].items() if p > 0}
            assert actual == pytest.approx(transformed.O[0][state])


def test_existing_revealing_signals_are_reused_across_actions():
    model = make_model(AtomicPropPOMDP, ("common", "known-left", "known-right"))
    model.addAtom(0, model.obs)
    model.setActions(["reveal", "mix"])
    model.T = {state: {action: {0: 0.4, 1: 0.6} for action in range(2)}
               for state in range(2)}
    model.O = {0: {0: {1: 1.0}, 1: {2: 1.0}}, 1: {0: {0: 1.0}, 1: {0: 1.0}}}
    model.transitions = {}
    model.computeTrans()
    transformed = make_strongly_revealing(model)
    assert transformed.obs == model.obs
    assert transformed.atoms == model.atoms
    assert transformed.O[0] == model.O[0]
    assert transformed.O[1] == {0: {0: 0.95, 1: 0.05}, 1: {0: 0.95, 2: 0.05}}
    assert is_strongly_revealing(transformed, preserve_labels=True)
    loaded = pomdp_parser.parse(transformed.to_pomdp_string())
    assert is_strongly_revealing(loaded, preserve_labels=True)


def test_reuse_accumulates_probability_already_on_the_revealing_signal():
    model = make_model(observations=("common", "known-left", "known-right"))
    model.T = {}
    model.O = {}
    model.transitions = {
        0: {0: {(0, 0): 0.4, (1, 0): 0.6}},
        1: {0: {(0, 0): 0.2, (0, 1): 0.3, (1, 0): 0.2, (1, 2): 0.3}},
    }
    transformed = make_strongly_revealing(model)
    assert transformed.obs == model.obs
    assert transformed.transitions[1][0][(0, 1)] == pytest.approx(0.3 + 0.05 * 0.2)
    assert transformed.transitions[1][0][(1, 2)] == pytest.approx(0.3 + 0.05 * 0.2)
    assert is_strongly_revealing(transformed)
    for state in range(2):
        actual = labelled_marginal(transformed, transformed.transitions[state][0])
        expected = labelled_marginal(model, model.transitions[state][0])
        assert actual == pytest.approx(expected)


def test_existing_signals_with_action_dependent_destinations_are_not_reused():
    model = make_model(observations=("common", "known"))
    model.setActions(["a", "b"])
    model.T = {state: {action: {0: 0.5, 1: 0.5} for action in range(2)}
               for state in range(2)}
    model.O = {0: {0: {0: 0.5, 1: 0.5}, 1: {0: 1.0}},
               1: {0: {0: 1.0}, 1: {0: 0.5, 1: 0.5}}}
    model.transitions = {}
    model.computeTrans()
    transformed = make_strongly_revealing(model)
    assert len(transformed.obs) == 4
    assert transformed.O[0][0] == model.O[0][0]
    assert transformed.O[1][1] == model.O[1][1]
    assert is_strongly_revealing(transformed)


def test_transformation_also_handles_joint_transitions_without_factorization():
    model = make_model(AtomicPropPOMDP, ("common", "left-only", "right-only"))
    model.T = {}
    model.O = {}
    # The observations depend on the source as well as the destination.
    model.transitions = {
        0: {0: {(0, 0): 1.0}},
        1: {0: {(0, 1): 0.5, (1, 0): 0.25, (1, 2): 0.25}},
    }
    transformed = make_strongly_revealing(model)
    assert is_strongly_revealing(transformed, preserve_labels=True)
    assert all(
        sum(actions[0].values()) == pytest.approx(1.0)
        for actions in transformed.transitions.values()
    )
    for state in range(2):
        actual = labelled_marginal(transformed, transformed.transitions[state][0])
        expected = labelled_marginal(model, model.transitions[state][0])
        assert actual == pytest.approx(expected)


def unnormalized_model(*, revealing):
    model = make_model(AtomicPropPOMDP, ("left-only", "right-only"))
    model.T = {0: {0: {0: 0.2, 1: 0.6}}, 1: {0: {0: 0.4, 1: 0.8}}}
    if revealing:
        model.O = {0: {0: {0: 1.0}, 1: {1: 1.0}}}
    model.transitions = {}
    model.computeTrans()
    return model


@pytest.mark.parametrize("factored", [True, False])
@pytest.mark.parametrize("revealing", [True, False])
def test_transformation_rejects_unnormalized_inputs(factored, revealing):
    model = unnormalized_model(revealing=revealing)
    if not factored:
        model.T = {}
        model.O = {}
    original = deepcopy(model.__dict__)
    with pytest.raises(ValueError, match="state 'left', action 'act'.*sum to 0.8"):
        make_strongly_revealing(model)
    assert model.__dict__ == original


@pytest.mark.parametrize("row", [
    {(0, 0): -0.2, (1, 0): 1.2},
    {(0, 0): float("nan")},
    {(0, 0): float("inf")},
    {(0, 0): 0.0},
    {(0, 0): 1.0 + 2e-12},
    {},
])
def test_transformation_rejects_invalid_joint_distributions(row):
    model = make_model()
    model.transitions[0][0] = row
    with pytest.raises(ValueError, match="state 'left', action 'act'"):
        make_strongly_revealing(model)


def test_transformation_rejects_missing_action_rows():
    model = make_model()
    del model.transitions[0][0]
    with pytest.raises(ValueError, match="sum to 0, expected 1"):
        make_strongly_revealing(model)


def test_validation_includes_observation_probabilities():
    model = make_model()
    model.O[0][0][0] = 0.8
    model.transitions = {}
    model.computeTrans()
    assert sum(model.T[0][0].values()) == 1.0
    with pytest.raises(ValueError, match="sum to 0.8, expected 1"):
        make_strongly_revealing(model)


def test_small_rounding_errors_are_accepted_without_normalization():
    model = make_model()
    model.T[0][0][0] = 1.0 + 5e-13
    model.transitions = {}
    model.computeTrans()
    original = deepcopy(model.__dict__)
    transformed = make_strongly_revealing(model)
    assert model.__dict__ == original
    assert transformed.T == model.T
    assert sum(transformed.transitions[0][0].values()) == sum(
        model.transitions[0][0].values()
    )


@pytest.mark.parametrize("entrypoint", ["cli", "batch-helper", "batch-script"])
def test_already_revealing_files_with_invalid_distributions_are_rejected(
    tmp_path, entrypoint, capsys
):
    source_dir = tmp_path / "examples" / "ltl"
    source_dir.mkdir(parents=True)
    source = source_dir / "input.pomdp"
    output = tmp_path / "output.pomdp"
    model = unnormalized_model(revealing=True)
    original = model.to_pomdp_string()
    source.write_text(original)
    if entrypoint == "batch-helper":
        from scripts.transform_to_revealing import transform_pomdp_file
        assert not transform_pomdp_file(source, output)
        assert "sum to 0.8, expected 1" in capsys.readouterr().out
    else:
        if entrypoint == "cli":
            command = [sys.executable, "-m", "pomdpy.revealing", "--file", str(source),
                       "--transform", "--output", str(output)]
            cwd = ROOT
        else:
            command = [sys.executable, str(ROOT / "scripts" / "transform_to_revealing.py")]
            cwd = tmp_path
            output = tmp_path / "examples" / "ltl-revealing" / "input.pomdp"
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=10)
        assert "sum to 0.8, expected 1" in result.stdout + result.stderr
        if entrypoint == "cli":
            assert result.returncode != 0
        else:
            assert "Failed: 1" in result.stdout
    assert source.read_text() == original
    assert not output.exists()


@pytest.mark.parametrize("epsilon", [0, 1, -0.1, float("inf"), float("nan")])
def test_invalid_revelation_fraction_is_rejected(epsilon):
    with pytest.raises(ValueError):
        make_strongly_revealing(make_model(), epsilon=epsilon)


@pytest.mark.parametrize("model_type", [POMDP, AtomicPropPOMDP, ParityPOMDP])
def test_export_preserves_small_probabilities_and_initial_distribution(model_type):
    model = make_model(model_type, ("rare", "common"))
    model.start = {0: 1 / 3, 1: 2 / 3}
    model.T = {state: {0: {0: 1 / 3, 1: 2 / 3}} for state in range(2)}
    model.O = {0: {state: {0: 1e-12, 1: 1 - 1e-12} for state in range(2)}}
    model.transitions = {}
    model.computeTrans()
    transformed = make_strongly_revealing(model)
    loaded = pomdp_parser.parse(transformed.to_pomdp_string())
    assert loaded.start == pytest.approx(transformed.start, rel=1e-13, abs=0)
    assert abs(sum(loaded.start.values()) - 1.0) < 1e-12
    assert is_strongly_revealing(loaded, preserve_labels=True)
    for state, actions in transformed.transitions.items():
        for action, outcomes in actions.items():
            expected = {key: prob for key, prob in outcomes.items() if prob > 0}
            actual = {key: prob for key, prob in loaded.transitions[state][action].items()
                      if prob > 0}
            assert actual.keys() == expected.keys()
            assert actual == pytest.approx(expected, rel=1e-13, abs=0)
            assert abs(sum(actual.values()) - 1.0) < 1e-12
    assert getattr(loaded, "atoms", None) == getattr(transformed, "atoms", None)
    assert getattr(loaded, "prio", None) == getattr(transformed, "prio", None)


def test_exported_uniform_initial_distribution_can_be_sampled():
    model = POMDP()
    model.setStates(["left", "middle", "right"])
    model.setActions(["act"])
    model.setObs(["common"])
    model.setUniformStart()
    model.addIdentityTrans()
    model.addUniformObs()
    loaded = pomdp_parser.parse(model.to_pomdp_string())
    assert loaded.start == pytest.approx(model.start, rel=1e-13, abs=0)
    loaded.reset()
    assert loaded.curstate in loaded.start


def test_cli_does_not_skip_a_strong_model_with_incompatible_labels(tmp_path):
    source = tmp_path / "input.pomdp"
    output = tmp_path / "output.pomdp"
    labelled_model().to_pomdp_file(source)
    result = subprocess.run(
        [sys.executable, "-m", "pomdpy.revealing", "--file", str(source),
         "--transform", "--output", str(output), "--epsilon", "0.2"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    transformed = pomdp_parser.parse(output.read_text())
    assert is_strongly_revealing(transformed, preserve_labels=True)
    assert transformed.transitions[0][0][(0, 0)] == pytest.approx(0.4)


def test_cli_rejects_invalid_epsilon_even_when_no_transformation_is_needed(tmp_path):
    source = tmp_path / "input.pomdp"
    model = labelled_model()
    model.atoms = {}
    model.to_pomdp_file(source)
    result = subprocess.run(
        [sys.executable, "-m", "pomdpy.revealing", "--file", str(source),
         "--transform", "--epsilon", "0"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "epsilon must be strictly between 0 and 1" in result.stderr


def test_batch_transformation_checks_the_labelling_condition(tmp_path):
    from scripts.transform_to_revealing import transform_pomdp_file

    source = tmp_path / "input.pomdp"
    output = tmp_path / "output.pomdp"
    source.write_text("# padding\n" * 3001 + labelled_model().to_pomdp_string())
    assert transform_pomdp_file(source, output)
    assert is_strongly_revealing(pomdp_parser.parse(output.read_text()), preserve_labels=True)


def test_batch_script_processes_files_above_the_former_line_cutoff(tmp_path):
    source_dir = tmp_path / "examples" / "ltl"
    source_dir.mkdir(parents=True)
    source = source_dir / "input.pomdp"
    source.write_text("# padding\n" * 3001 + labelled_model().to_pomdp_string())

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "transform_to_revealing.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Successfully transformed: 1" in result.stdout
    assert "Skipped (too large)" not in result.stdout
    output = tmp_path / "examples" / "ltl-revealing" / "input.pomdp"
    assert is_strongly_revealing(pomdp_parser.parse(output.read_text()), preserve_labels=True)
