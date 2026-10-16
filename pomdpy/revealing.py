from collections import defaultdict
from copy import deepcopy
from math import fsum, isclose, isfinite

from pomdpy.pomdp import AtomicPropPOMDP


def _missing_revelations(pomdp, preserve_labels):
    """Yield (action, destination, observation) outcomes lacking an alternative."""
    # A signal reveals its destination independently of the source state.
    destinations = defaultdict(set)
    for actions in pomdp.transitions.values():
        for action_id, outcomes in actions.items():
            for (next_state_id, obs_id), prob in outcomes.items():
                if prob > 0:
                    destinations[(action_id, obs_id)].add(next_state_id)
    revealing = {key for key, states in destinations.items() if len(states) == 1}

    atoms = getattr(pomdp, "atoms", {}) if preserve_labels else {}
    labels = {
        obs_id: frozenset(atom for atom, obs_set in atoms.items() if obs_id in obs_set)
        for obs_id in range(len(pomdp.obs))
    }
    for actions in pomdp.transitions.values():
        for action_id, outcomes in actions.items():
            # Alternatives must occur on the same source/action/destination edge.
            covered = {(next_state_id, labels[obs_id])
                       for (next_state_id, obs_id), prob in outcomes.items()
                       if prob > 0 and (action_id, obs_id) in revealing}
            for (next_state_id, obs_id), prob in outcomes.items():
                if prob > 0 and (next_state_id, labels[obs_id]) not in covered:
                    yield action_id, next_state_id, obs_id


def is_strongly_revealing(pomdp, *, preserve_labels=False):
    """
    Check if a POMDP is strongly revealing (Definition 5.1).

    Every possible source/action/destination edge must admit a positive signal
    identifying the destination across all source states for that action.
    This checks the positive transition support, assuming valid distributions.

    Args:
        pomdp: A POMDP object
        preserve_labels: Also require each outcome's revealing alternative to
            have the same atomic propositions (condition (2)). Without atomic
            propositions, this is the same as ordinary strong revelation.

    Returns:
        bool: True if the required revealing alternatives exist, False otherwise
    """
    for _ in _missing_revelations(pomdp, preserve_labels):
        return False
    return True


def validate_transition_rows(pomdp):
    """Check joint next-state/observation distributions without changing them."""
    for state_id, state in enumerate(pomdp.states):
        for action_id, action in enumerate(pomdp.actions):
            row = pomdp.transitions.get(state_id, {}).get(action_id, {})
            if any(not isfinite(p) or p < 0 for p in row.values()):
                raise ValueError(
                    f"Invalid transition probabilities for state {state!r}, action {action!r}"
                )
            total = fsum(row.values())
            if not isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"Transition probabilities for state {state!r}, action {action!r} "
                    f"sum to {total:.14g}, expected 1"
                )


def make_strongly_revealing(pomdp, *, epsilon=0.05):
    """
    Transform a POMDP into a strongly revealing one by adding revealing observations.

    For outcomes without a suitable revealing alternative, split probability p
    into (1 - epsilon)*p on the original signal and epsilon*p on a revealing
    signal for the same destination and atomic propositions (condition (2)).
    Reuse compatible existing signals, and share new signals between outcomes
    with the same destination and label.

    The input is not modified. Invalid transition distributions are rejected.
    State dynamics, label distributions, priorities and the initial distribution
    are preserved.
    T/O tables, when present, remain consistent with the joint transitions.

    Args:
        pomdp: A POMDP object
        epsilon: Fraction of each affected outcome assigned to its revealing
            signal (default: 0.05). Must be strictly between zero and one.

    Returns:
        POMDP: An independent strongly revealing copy (same type as input)
    """
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be strictly between 0 and 1")

    validate_transition_rows(pomdp)
    missing = set(_missing_revelations(pomdp, isinstance(pomdp, AtomicPropPOMDP)))
    # Transition keys and probabilities are immutable; copy their dictionaries
    # directly instead of recursively copying every outcome, including zeros.
    transitions = {
        state: {action: outcomes.copy() for action, outcomes in actions.items()}
        for state, actions in pomdp.transitions.items()
    }
    new_pomdp = deepcopy(pomdp, {id(pomdp.transitions): transitions})
    if not missing:
        return new_pomdp
    atoms = pomdp.atoms if isinstance(pomdp, AtomicPropPOMDP) else {}
    labels = {
        obs_id: frozenset(atom for atom, obs_set in atoms.items() if obs_id in obs_set)
        for obs_id in range(len(pomdp.obs))
    }

    # Reuse only signals identifying the same destination across all actions.
    destinations = defaultdict(set)
    for actions in pomdp.transitions.values():
        for outcomes in actions.values():
            for (next_state_id, obs_id), prob in outcomes.items():
                if prob > 0:
                    destinations[obs_id].add(next_state_id)
    copies = {}
    for obs_id in sorted(destinations):
        states = destinations[obs_id]
        if len(states) == 1:
            copies.setdefault((next(iter(states)), labels[obs_id]), obs_id)

    for action_id, next_state_id, obs_id in sorted(missing):
        key = (next_state_id, labels[obs_id])
        if key in copies:
            continue
        base_name = f"reveal_{pomdp.states[next_state_id]}"
        if labels[obs_id]:
            base_name += "_" + "_".join(f"p{atom}" for atom in sorted(labels[obs_id]))
        reveal_obs = base_name
        suffix = 1
        while reveal_obs in new_pomdp.obsinv:
            reveal_obs = f"{base_name}_{suffix}"
            suffix += 1
        new_obs_id = len(new_pomdp.obs)
        copies[key] = new_obs_id
        new_pomdp.obs.append(reveal_obs)
        new_pomdp.obsinv[reveal_obs] = new_obs_id
        for atom in labels[obs_id]:
            new_pomdp.atoms[atom].add(new_obs_id)

    # Use the same split across sources to preserve any existing T/O factorization.
    for state_id, actions in pomdp.transitions.items():
        for action_id, outcomes in actions.items():
            new_outcomes = new_pomdp.transitions[state_id][action_id]
            for (next_state_id, obs_id), prob in outcomes.items():
                if prob > 0 and (action_id, next_state_id, obs_id) in missing:
                    new_obs_id = copies[(next_state_id, labels[obs_id])]
                    new_outcomes[(next_state_id, obs_id)] = (1 - epsilon) * prob
                    key = (next_state_id, new_obs_id)
                    new_outcomes[key] = new_outcomes.get(key, 0.0) + epsilon * prob

    for action_id, next_state_id, obs_id in sorted(missing):
        row = new_pomdp.O.get(action_id, {}).get(next_state_id, {})
        if obs_id in row:
            prob = row[obs_id]
            row[obs_id] = (1 - epsilon) * prob
            new_obs_id = copies[(next_state_id, labels[obs_id])]
            row[new_obs_id] = row.get(new_obs_id, 0.0) + epsilon * prob

    return new_pomdp


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Add parent directory to path to allow running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pomdpy.parsers import pomdp as pomdp_parser
    
    parser = argparse.ArgumentParser(
        description="Check if a POMDP is strongly revealing and optionally transform it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if a POMDP is revealing
  python -m pomdpy.revealing --file examples/ltl-tiger.pomdp
  
  # Transform a POMDP to be strongly revealing
  python -m pomdpy.revealing --file examples/ltl-tiger.pomdp --transform
  
  # Transform and save to a specific file
  python -m pomdpy.revealing --file examples/ltl-tiger.pomdp --transform --output examples/revealing_tiger.pomdp
  
  # Just transform without checking (faster)
  python -m pomdpy.revealing --file examples/ltl-tiger.pomdp --transform --no-check
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the POMDP file"
    )
    
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Transform the POMDP to be strongly revealing"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for transformed POMDP (default: add 'revealing_' prefix)"
    )
    
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip checking if POMDP is already revealing (useful for faster transformation)"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Fraction of each affected outcome assigned to its revealing signal (default: 0.05)"
    )
    
    args = parser.parse_args()

    if not 0 < args.epsilon < 1:
        parser.error("epsilon must be strictly between 0 and 1")
    
    # Load POMDP
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    
    print(f"Loading POMDP from {args.file}...")
    with open(input_path, 'r') as f:
        pomdp = pomdp_parser.parse(f.read())

    try:
        validate_transition_rows(pomdp)
    except ValueError as e:
        parser.error(str(e))
    
    print(f"Loaded: {type(pomdp).__name__}")
    print(f"  States: {len(pomdp.states)}")
    print(f"  Actions: {len(pomdp.actions)}")
    print(f"  Observations: {len(pomdp.obs)}")
    if isinstance(pomdp, AtomicPropPOMDP) and pomdp.atoms:
        print(f"  Atomic propositions: {list(pomdp.atoms.keys())}")
    
    # Check if revealing (unless --no-check is specified)
    is_revealing = None
    if not args.no_check:
        print("\nChecking if POMDP is strongly revealing...")
        is_revealing = is_strongly_revealing(pomdp)
        if is_revealing:
            print("Result: POMDP is strongly revealing")
        else:
            print("Result: POMDP is NOT strongly revealing")
    
    # Transform if requested
    if args.transform:
        if is_revealing and is_strongly_revealing(pomdp, preserve_labels=True):
            print("\nPOMDP already has label-preserving revealing alternatives. Skipping transformation.")
            if args.output:
                print(f"Copying to {args.output}...")
                import shutil
                shutil.copy2(input_path, args.output)
                print("Done.")
        else:
            print("\nTransforming to strongly revealing POMDP...")
            revealing_pomdp = make_strongly_revealing(pomdp, epsilon=args.epsilon)
            print("Transformed POMDP:")
            print(f"  States: {len(revealing_pomdp.states)}")
            print(f"  Actions: {len(revealing_pomdp.actions)}")
            print(f"  Observations: {len(revealing_pomdp.obs)}")
            
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = input_path.parent / f"revealing_{input_path.name}"
            
            print(f"\nWriting to {output_path}...")
            revealing_pomdp.to_pomdp_file(str(output_path))
            print("Done.")
            
            # Verify transformation
            print("\nVerifying transformation...")
            if is_strongly_revealing(revealing_pomdp, preserve_labels=True):
                print("Verification: Transformed POMDP has label-preserving revealing alternatives")
            else:
                print("Warning: Transformed POMDP lacks label-preserving revealing alternatives")
    elif not args.no_check and is_revealing is False:
        print("\nTo transform this POMDP, run with --transform option")
    
    sys.exit(0)
