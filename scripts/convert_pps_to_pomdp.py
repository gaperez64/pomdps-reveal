#!/usr/bin/env python3
"""
Convert PPS (Parity POMDP Solver) format files to standard POMDP format.

PPS Format specification:
- NAME: [name] #
- OBSERVATIONS: [obs1, obs2, ...] #
- STATES: [{state1;priority;observation;isTarget}, ...] #
- ACTIONS: [action1, action2, ...] #
- TRANSITIONS: [state,action,{succ1; succ2; ...}] #

The states include priority information (for parity objectives) and target flags (T/F).
Each state is associated with an observation.
Transitions are non-deterministic but uniform (equal probability to all successors).
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from pomdpy.pomdp import ParityPOMDP


def parse_pps_file(filepath):
    """
    Parse a PPS format file and return structured data.
    
    Returns:
        dict with keys: name, observations, states, actions, transitions
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {}
    
    # Parse NAME
    name_match = re.search(r'NAME:\s*\[(.*?)\]\s*#', content)
    if name_match:
        data['name'] = name_match.group(1).strip()
    else:
        data['name'] = filepath.stem
    
    # Parse OBSERVATIONS
    obs_match = re.search(r'OBSERVATIONS:\s*\[(.*?)\]\s*#', content)
    if obs_match:
        obs_str = obs_match.group(1)
        # Split by comma, handling potential whitespace
        data['observations'] = [o.strip() for o in obs_str.split(',')]
    else:
        raise ValueError("No OBSERVATIONS found in PPS file")
    
    # Parse STATES
    # Format: {state_name; priority; observation; target_flag}
    states_match = re.search(r'STATES:\s*\[(.*?)\]\s*#', content, re.DOTALL)
    if states_match:
        states_str = states_match.group(1)
        # Extract each state block {....}
        state_pattern = r'\{([^}]+)\}'
        state_matches = re.findall(state_pattern, states_str)
        
        data['states'] = []
        for state_str in state_matches:
            parts = [p.strip() for p in state_str.split(';')]
            if len(parts) >= 4:
                state_info = {
                    'name': parts[0],
                    'priority': int(parts[1]),
                    'observation': parts[2],
                    'is_target': parts[3] == 'T'
                }
                data['states'].append(state_info)
    else:
        raise ValueError("No STATES found in PPS file")
    
    # Parse ACTIONS
    actions_match = re.search(r'ACTIONS:\s*\[(.*?)\]\s*#', content)
    if actions_match:
        actions_str = actions_match.group(1)
        data['actions'] = [a.strip() for a in actions_str.split(',')]
    else:
        raise ValueError("No ACTIONS found in PPS file")
    
    # Parse TRANSITIONS
    # Format: TRANSITIONS: [state,action,{succ1; succ2; ...}] #
    trans_pattern = r'TRANSITIONS:\s*\[([^,]+),([^,]+),\{([^}]+)\}\]\s*#'
    trans_matches = re.findall(trans_pattern, content)
    
    data['transitions'] = []
    for src, act, succs_str in trans_matches:
        src = src.strip()
        act = act.strip()
        # Filter out empty strings from successors
        successors = [s.strip() for s in succs_str.split(';') if s.strip()]
        if successors:  # Only add if there are valid successors
            data['transitions'].append({
                'source': src,
                'action': act,
                'successors': successors
            })
    
    return data


def pps_to_pomdp(pps_data):
    """
    Convert parsed PPS data to a ParityPOMDP object.
    
    Args:
        pps_data: Dictionary from parse_pps_file
        
    Returns:
        ParityPOMDP object
    """
    pomdp = ParityPOMDP()
    
    # Set states (just the names)
    # Prefix numeric state names with 's' to match POMDP parser requirements
    state_names = [f"s{s['name']}" for s in pps_data['states']]
    pomdp.setStates(state_names)
    
    # Create a mapping from original PPS state name to new prefixed name
    pps_state_to_pomdp_state = {s['name']: f"s{s['name']}" for s in pps_data['states']}
    
    # Set actions
    pomdp.setActions(pps_data['actions'])
    
    # Set observations
    pomdp.setObs(pps_data['observations'])
    
    # Set priorities for states (group by priority)
    priorities_map = defaultdict(list)  # priority -> list of state names
    for s in pps_data['states']:
        priorities_map[s['priority']].append(pps_state_to_pomdp_state[s['name']])
    
    for priority, state_list in priorities_map.items():
        pomdp.addPriority(priority, state_list)
    
    # Build state name to observation mapping (use prefixed names)
    state_to_obs = {pps_state_to_pomdp_state[s['name']]: s['observation'] for s in pps_data['states']}
    
    # Set start distribution
    # Find states with is_target=True as starting states
    start_states = [i for i, s in enumerate(pps_data['states']) if s['is_target']]
    
    if not start_states:
        # If no target states marked for start, use first state
        print(f"Warning: No target starting states found, using first state")
        start_states = [0]
    
    # Uniform distribution over start states
    for state_id in start_states:
        pomdp.start[state_id] = 1.0 / len(start_states)
    
    # Build transition function T
    # Group transitions by (source_state, action)
    trans_map = defaultdict(list)
    for trans in pps_data['transitions']:
        src_name = pps_state_to_pomdp_state[trans['source']]
        src_id = pomdp.statesinv[src_name]
        act_id = pomdp.actionsinv[trans['action']]
        succ_ids = [pomdp.statesinv[pps_state_to_pomdp_state[s]] for s in trans['successors'] if s.strip()]
        trans_map[(src_id, act_id)] = succ_ids
    
    # Add transitions with uniform probability
    for (src_id, act_id), succ_ids in trans_map.items():
        if src_id not in pomdp.T:
            pomdp.T[src_id] = {}
        if act_id not in pomdp.T[src_id]:
            pomdp.T[src_id][act_id] = {}
        
        prob = 1.0 / len(succ_ids)
        for succ_id in succ_ids:
            pomdp.T[src_id][act_id][succ_id] = prob
    
    # Build observation function O
    # Each state has a deterministic observation
    for act_id in range(len(pomdp.actions)):
        if act_id not in pomdp.O:
            pomdp.O[act_id] = {}
        
        for state_id, state_name in enumerate(pomdp.states):
            if state_id not in pomdp.O[act_id]:
                pomdp.O[act_id][state_id] = {}
            
            # Get the observation for this state
            obs_name = state_to_obs[state_name]
            obs_id = pomdp.obsinv[obs_name]
            
            # Deterministic observation
            pomdp.O[act_id][state_id][obs_id] = 1.0
    
    # Build transitions directly from T and O (avoid expensive computeTrans)
    # For each state-action pair, multiply T and O probabilities
    for state_id in range(len(pomdp.states)):
        pomdp.transitions[state_id] = {}
        for action_id in range(len(pomdp.actions)):
            pomdp.transitions[state_id][action_id] = {}
            
            if state_id in pomdp.T and action_id in pomdp.T[state_id]:
                # For each next state
                for next_state_id, t_prob in pomdp.T[state_id][action_id].items():
                    if t_prob > 0:
                        # Get observation for next state
                        if action_id in pomdp.O and next_state_id in pomdp.O[action_id]:
                            for obs_id, o_prob in pomdp.O[action_id][next_state_id].items():
                                # Joint probability
                                joint_prob = t_prob * o_prob
                                if joint_prob > 0:
                                    pomdp.transitions[state_id][action_id][(next_state_id, obs_id)] = joint_prob
    
    return pomdp


def convert_pps_file(input_path, output_path=None):
    """
    Convert a single PPS file to POMDP format.
    
    Args:
        input_path: Path to input PPS file
        output_path: Path to output POMDP file (optional)
        
    Returns:
        Path to output file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix('.pomdp')
    else:
        output_path = Path(output_path)
    
    print(f"Converting {input_path.name}...")
    
    try:
        # Parse PPS file
        pps_data = parse_pps_file(input_path)
        
        print(f"  States: {len(pps_data['states'])}")
        print(f"  Actions: {len(pps_data['actions'])}")
        print(f"  Observations: {len(pps_data['observations'])}")
        print(f"  Transitions: {len(pps_data['transitions'])}")
        
        # Convert to POMDP
        pomdp = pps_to_pomdp(pps_data)
        
        # Write to file using to_pomdp_file, but with custom format
        # since the standard matrix format isn't well-supported by the parser
        lines = []
        
        # Write header
        lines.append("# POMDP file converted from PPS format")
        lines.append("")
        
        # Write states
        lines.append(f"states: {' '.join(pomdp.states)}")
        
        # Write actions
        lines.append(f"actions: {' '.join(pomdp.actions)}")
        
        # Write observations
        lines.append(f"observations: {' '.join(pomdp.obs)}")
        lines.append("")
        
        # Write priorities (prio declarations)
        for priority in sorted(pomdp.prio.keys()):
            state_names = [pomdp.states[state_id] for state_id in sorted(pomdp.prio[priority])]
            lines.append(f"prio {priority}: {' '.join(state_names)}")
        lines.append("")
        
        # Write start distribution using include syntax
        start_states = [pomdp.states[state_id] for state_id in range(len(pomdp.states)) if pomdp.start.get(state_id, 0.0) > 0]
        if start_states:
            lines.append(f"start include: {' '.join(start_states)}")
        else:
            lines.append(f"start include: {pomdp.states[0]}")  # Fallback to first state
        lines.append("")
        
        # Write transitions using T: format with one row per line
        for action_id in range(len(pomdp.actions)):
            lines.append(f"T:{pomdp.actions[action_id]}")
            
            # Write one row per state for this action
            for state_id in range(len(pomdp.states)):
                row = []
                for next_state_id in range(len(pomdp.states)):
                    prob = 0.0
                    if state_id in pomdp.T and action_id in pomdp.T[state_id]:
                        prob = pomdp.T[state_id][action_id].get(next_state_id, 0.0)
                    row.append(f"{prob:.2f}")
                lines.append(" ".join(row))
            lines.append("")
        
        # Write observations using O: format with one row per line
        for action_id in range(len(pomdp.actions)):
            lines.append(f"O:{pomdp.actions[action_id]}")
            
            # Write one row per state for this action
            for next_state_id in range(len(pomdp.states)):
                row = []
                for obs_id in range(len(pomdp.obs)):
                    prob = 0.0
                    if action_id in pomdp.O and next_state_id in pomdp.O[action_id]:
                        prob = pomdp.O[action_id][next_state_id].get(obs_id, 0.0)
                    row.append(f"{prob:.2f}")
                lines.append(" ".join(row))
            lines.append("")
        
        content = "\n".join(lines)
        with open(str(output_path), 'w') as f:
            f.write(content)
        
        return output_path
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_pps_directory(input_dir, output_dir=None, recursive=True):
    """
    Convert all PPS files in a directory.
    
    Args:
        input_dir: Directory containing PPS files
        output_dir: Output directory (default: same as input)
        recursive: Whether to process subdirectories
    """
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files
    if recursive:
        pps_files = list(input_dir.rglob('*.txt'))
    else:
        pps_files = list(input_dir.glob('*.txt'))
    
    # Filter out files that don't look like PPS format
    # (simple heuristic: must contain OBSERVATIONS, STATES, ACTIONS, TRANSITIONS)
    valid_files = []
    for f in pps_files:
        try:
            content = f.read_text()
            if all(keyword in content for keyword in ['OBSERVATIONS:', 'STATES:', 'ACTIONS:', 'TRANSITIONS:']):
                valid_files.append(f)
        except:
            pass
    
    print(f"Found {len(valid_files)} PPS files in {input_dir}")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for pps_file in sorted(valid_files):
        # Maintain directory structure
        if recursive:
            rel_path = pps_file.relative_to(input_dir)
            out_file = output_dir / rel_path.with_suffix('.pomdp')
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_file = output_dir / pps_file.with_suffix('.pomdp').name
        
        result = convert_pps_file(pps_file, out_file)
        if result:
            success_count += 1
        else:
            fail_count += 1
        print()
    
    print("=" * 60)
    print(f"Conversion complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert PPS format files to standard POMDP format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python scripts/convert_pps_to_pomdp.py --file examples/pps/4x3_grid/4x3_grid.txt
  
  # Convert all PPS files in a directory
  python scripts/convert_pps_to_pomdp.py --dir examples/pps
  
  # Convert with custom output directory
  python scripts/convert_pps_to_pomdp.py --dir examples/pps --output examples/parity
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Convert a single PPS file'
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        help='Convert all PPS files in a directory (recursive)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory or file path'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not process subdirectories when using --dir'
    )
    
    args = parser.parse_args()
    
    if args.file:
        convert_pps_file(args.file, args.output)
    elif args.dir:
        convert_pps_directory(args.dir, args.output, not args.no_recursive)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
