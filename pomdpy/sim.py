#!/usr/bin/python3

import argparse

from pomdpy.parsers import pomdp
from pomdpy.pomdp import POMDP, ParityPOMDP, AtomicPropPOMDP


def print_pomdp_info(env: POMDP):
    """Print information about the POMDP type and its specifications."""
    print(f"\n{'='*60}")
    print(f"POMDP Type: {type(env).__name__}")
    print(f"{'='*60}")
    print(f"States: {len(env.states)} - {env.states}")
    print(f"Actions: {len(env.actions)} - {env.actions}")
    print(f"Observations: {len(env.obs)} - {env.obs}")
    
    # Display type-specific information
    if isinstance(env, ParityPOMDP):
        print("\nPriority Function (state-based):")
        if env.prio:
            for priority in sorted(env.prio.keys()):
                state_names = [env.states[s]
                              for s in sorted(env.prio[priority])]
                print(f"  Priority {priority}: {state_names}")
        else:
            print("  No priorities defined")
    
    elif isinstance(env, AtomicPropPOMDP):
        print("\nAtomic Propositions (observation-based):")
        if env.atoms:
            for atom_id in sorted(env.atoms.keys()):
                obs_names = [env.obs[o] for o in sorted(env.atoms[atom_id])]
                print(f"  p{atom_id}: {obs_names}")
        else:
            print("  No atomic propositions defined")
    
    print("\nInitial distribution:")
    if env.start:
        for state_id in sorted(env.start.keys()):
            print(f"  {env.states[state_id]}: {env.start[state_id]:.3f}")
    else:
        print("  Not defined")
    print(f"{'='*60}\n")


def update_belief(env: POMDP, belief: dict, action_idx: int, observation_idx: int) -> dict:
    """
    Update belief state after taking an action and receiving an observation.
    
    Args:
        env: POMDP environment
        belief: Current belief (dict mapping state indices to probabilities)
        action_idx: Index of action taken
        observation_idx: Index of observation received
    
    Returns:
        Updated belief (normalized probability distribution)
    """
    new_belief = {}
    
    # For each possible next state
    for next_state in range(len(env.states)):
        prob_sum = 0.0
        
        # Sum over all current states weighted by belief
        for curr_state, curr_prob in belief.items():
            if curr_prob == 0:
                continue
                
            # Check if transition exists
            if curr_state in env.transitions and \
               action_idx in env.transitions[curr_state]:
                trans = env.transitions[curr_state][action_idx]
                
                # Get probability of (next_state, observation)
                if (next_state, observation_idx) in trans:
                    prob_sum += curr_prob * trans[(next_state, observation_idx)]
        
        if prob_sum > 0:
            new_belief[next_state] = prob_sum
    
    # Normalize
    total = sum(new_belief.values())
    if total > 0:
        new_belief = {s: p/total for s, p in new_belief.items()}
    
    return new_belief


def print_belief(env: POMDP, belief: dict, threshold: float = 0.001):
    """
    Print belief state, showing only states with probability above threshold.
    
    Args:
        env: POMDP environment
        belief: Belief state (dict mapping state indices to probabilities)
        threshold: Minimum probability to display (default: 0.001 = 0.1%)
    """
    print("  Belief state:")
    sorted_belief = sorted(belief.items(), key=lambda x: x[1], reverse=True)
    
    displayed_prob = 0.0
    for state_idx, prob in sorted_belief:
        if prob >= threshold:
            state_name = env.states[state_idx]
            print(f"    {state_name}: {prob:.4f}")
            displayed_prob += prob
    
    # Show if some probability mass was hidden
    if displayed_prob < 0.999:
        print(f"    (... {1-displayed_prob:.4f} in other states)")


def simulate(filename: str, verbose: bool = False):
    """
    Simulate a POMDP interactively, maintaining belief state.
    
    Args:
        filename: Path to POMDP file
        verbose: If True, print detailed transition information
    """
    # Parse the POMDP file
    with open(filename) as f:
        env = pomdp.parse(f.read())
        
    # Display POMDP information
    print_pomdp_info(env)
    
    # Initialize belief from start distribution
    belief = dict(env.start)
    
    # Reset to initial state (for the actual hidden state)
    env.reset()
    current_state = env.states[env.curstate]
    
    print(f"Hidden true state: {current_state}")
    print_belief(env, belief)
    
    # Display state-specific info if applicable
    if isinstance(env, ParityPOMDP) and env.prio:
        for priority, states in env.prio.items():
            if env.curstate in states:
                print(f"  (Priority {priority})")
    
    prev_action = None
    step_count = 0
    
    print("\nSimulation started. Press Ctrl+C to exit.")
    print("Enter an action name, or press Enter to repeat the last action.\n")
    
    try:
        while True:
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            print(f"Hidden true state: {current_state} (index: {env.curstate})")
            print_belief(env, belief)

            # Display state-specific info
            if isinstance(env, ParityPOMDP) and env.prio:
                for priority, states in env.prio.items():
                    if env.curstate in states:
                        print(f"  True state priority: {priority}")
            
            # Display available transitions if verbose
            if verbose and env.curstate in env.transitions:
                print("\nAvailable transitions from true state:")
                for act_idx, act_name in enumerate(env.actions):
                    if act_idx in env.transitions[env.curstate]:
                        trans_dict = env.transitions[env.curstate][act_idx]
                        if trans_dict:
                            print(f"  {act_name}:")
                            items = sorted(trans_dict.items())
                            for (next_s, obs), prob in items:
                                if prob > 0:
                                    next_state = env.states[next_s]
                                    obs_name = env.obs[obs]
                                    print(f"    -> {next_state}, "
                                          f"obs={obs_name}: p={prob:.3f}")
            
            print(f"\nEnter action {env.actions}: ", end='')
            act = input().strip()
            
            # Handle empty input (repeat last action)
            if act == "":
                if prev_action is not None:
                    act = prev_action
                    print(f"Repeating: {act}")
                else:
                    print("No previous action. Please enter an action.")
                    continue
            
            # Validate action
            if act not in env.actions:
                print(f"Invalid action '{act}'. "
                      f"Please choose from {env.actions}")
                continue
            
            prev_action = act
            act_idx = env.actionsinv[act]
            
            # Take step in actual environment
            observation = env.step(act_idx)
            obs_idx = env.obsinv[observation]
            current_state = env.states[env.curstate]
            
            # Update belief
            belief = update_belief(env, belief, act_idx, obs_idx)
            
            # Display results
            print(f"\n  Action taken: {act}")
            print(f"  Observation: {observation}")
            print(f"  Hidden true state: {current_state} (index: {env.curstate})")
            print_belief(env, belief)
                        
            # Display observation-specific info
            if isinstance(env, AtomicPropPOMDP) and env.atoms:
                atoms_true = [f"p{atom}" for atom, obs_set in env.atoms.items()
                             if obs_idx in obs_set]
                if atoms_true:
                    print(f"  Atomic propositions true: {atoms_true}")
    
    except KeyboardInterrupt:
        print(f"\n\nSimulation ended after {step_count} steps.")
        print(f"Hidden true state: {current_state}")
        print_belief(env, belief)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sim.py",
        description="Simulates a POMDP with support for ParityPOMDP and AtomicPropPOMDP"
    )
    parser.add_argument(
        "filename",
        help="POMDP filename"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed transition information at each step"
    )

    args = parser.parse_args()
    simulate(args.filename, args.verbose)
    exit(0)
