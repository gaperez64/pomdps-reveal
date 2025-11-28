from pomdpy.pomdp import POMDP, AtomicPropPOMDP
from collections import defaultdict

def is_strongly_revealing(pomdp):
    """
    Check if a POMDP is strongly revealing.
    
    A POMDP is strongly revealing if for any state q and q', for any action a,
    if it is possible to go to q' from q with action a, then there is a revealing
    observation o that reveals q' from q with action a.
        
    Args:
        pomdp: A POMDP object
        
    Returns:
        bool: True if the POMDP is strongly revealing, False otherwise
    """
    
    for state_id in range(len(pomdp.states)):            
        for action_id in range(len(pomdp.actions)):            
            obs_next_states = defaultdict(set)
            for (next_state_id, obs_id), prob in pomdp.transitions[state_id][action_id].items():
                if prob > 0:
                    obs_next_states[obs_id].add(next_state_id)
            
            # Get all unique next states reachable from this state-action
            all_next_states = set()
            for next_states_set in obs_next_states.values():
                all_next_states.update(next_states_set)

            # For each reachable next state, check if there's a revealing observation
            for next_state_id in all_next_states:
                has_revealing_obs = False
                
                # Check all observations from (state_id, action_id)
                for obs_id, next_states in obs_next_states.items():
                    # Check if this observation reveals next_state_id uniquely
                    if len(next_states) == 1 and next_state_id in next_states:
                        has_revealing_obs = True
                        break
                
                if not has_revealing_obs:
                    print(f"No observation from state {state_id} to state {next_state_id} with action {action_id}")
                    return False
    
    return True


def make_strongly_revealing(pomdp):
    """
    Transform a POMDP into a strongly revealing one by adding revealing observations.
    
    For each state q', we add a revealing observation o_q'. For each transition
    (q, a, o, q') that exists with positive probability, we add a transition (q, a, o_q', q').
    We remove a small epsilon from probability of (q, a, o, q') and redistribute it to the new transition (q, a, o_q', q').
    
    Args:
        pomdp: A POMDP object
        
    Returns:
        POMDP: A new strongly revealing POMDP (same type as input)
    """
    epsilon = 0.05  # Small probability to transfer to revealing observations
    
    # Create a new POMDP of the same type as the input
    if isinstance(pomdp, AtomicPropPOMDP):
        new_pomdp = AtomicPropPOMDP()
    else:
        new_pomdp = POMDP()
    
    # Copy basic attributes
    new_pomdp.states = pomdp.states.copy()
    new_pomdp.actions = pomdp.actions.copy()
    new_pomdp.statesinv = pomdp.statesinv.copy()
    new_pomdp.actionsinv = pomdp.actionsinv.copy()
            
    # Copy start distribution
    new_pomdp.start = pomdp.start.copy()
    
    new_pomdp.T = pomdp.T.copy()
    new_pomdp.transitions = {}

    # Build O (observation function)
    new_pomdp.O = pomdp.O.copy()

    # Start from existing observations
    new_pomdp.obs = pomdp.obs.copy()
    new_pomdp.obsinv = {obs: idx for idx, obs in enumerate(new_pomdp.obs)}
    
    # Copy atomic propositions if present
    if isinstance(pomdp, AtomicPropPOMDP):
        new_pomdp.atoms = {
            atom: obs_set.copy()
            for atom, obs_set in pomdp.atoms.items()
        }

    for state_id in range(len(pomdp.states)):            
        for action_id in range(len(pomdp.actions)):            
            obs_next_states = defaultdict(set)
            for (next_state_id, obs_id), prob in pomdp.transitions[state_id][action_id].items():
                if prob > 0:
                    obs_next_states[obs_id].add(next_state_id)            

            # Get all unique next states reachable from this state-action
            all_next_states = set()
            for next_states_set in obs_next_states.values():
                all_next_states.update(next_states_set)

            # For each reachable next state, check if there's a revealing observation
            for next_state_id in all_next_states:
                # print(f"Considering state {next_state_id}")
                has_revealing_obs = False
                
                # Check all observations from (state_id, action_id)
                for obs_id, next_states in obs_next_states.items():
                    # Check if this observation reveals next_state_id uniquely
                    if len(next_states) == 1 and next_state_id in next_states:
                        has_revealing_obs = True
                        break
                
                if not has_revealing_obs:
                    # Need to add revealing observation
                    reveal_obs = f"reveal_{new_pomdp.states[next_state_id]}"
                    if reveal_obs not in new_pomdp.obsinv:
                        new_obs_id = len(new_pomdp.obs)
                        new_pomdp.obs.append(reveal_obs)
                        new_pomdp.obsinv[reveal_obs] = new_obs_id
                        
                        # For AtomicPropPOMDP, revealing obs don't satisfy any atoms
                        # (atoms remain as they were, no new obs added to them)
                    else:
                        new_obs_id = new_pomdp.obsinv[reveal_obs]
                                        
                    # Add the revealing transition
                    new_pomdp.O[action_id][next_state_id][new_obs_id] = epsilon

    for action_id in range(len(pomdp.actions)):
        for next_state_id in range(len(pomdp.states)):
            # Renormalize observation probabilities for (state_id, action_id)
            s = sum(new_pomdp.O[action_id][next_state_id].values())
            if s > 0:
                for obs_id in pomdp.O[action_id][next_state_id].keys():
                    new_pomdp.O[action_id][next_state_id][obs_id] /= s

    new_pomdp.computeTrans()
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
    
    args = parser.parse_args()
    
    # Load POMDP
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    
    print(f"Loading POMDP from {args.file}...")
    with open(input_path, 'r') as f:
        pomdp = pomdp_parser.parse(f.read())
    
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
        if is_revealing and not args.no_check:
            print("\nPOMDP is already strongly revealing. Skipping transformation.")
            if args.output:
                print(f"Copying to {args.output}...")
                import shutil
                shutil.copy2(input_path, args.output)
                print("Done.")
        else:
            print("\nTransforming to strongly revealing POMDP...")
            revealing_pomdp = make_strongly_revealing(pomdp)
            print(f"Transformed POMDP:")
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
            if is_strongly_revealing(revealing_pomdp):
                print("Verification: Transformed POMDP is strongly revealing")
            else:
                print("Warning: Transformed POMDP is still not strongly revealing")
    elif not args.no_check and is_revealing is False:
        print("\nTo transform this POMDP, run with --transform option")
    
    sys.exit(0)
