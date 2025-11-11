import argparse
import os
from pomdpy.pomdp import AtomicPropPOMDP
from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
import spot

try:
    import pygraphviz as pgv
    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False


def render_dot_to_png(dot_path, png_path):
    """
    Render a DOT file to PNG using pygraphviz if available.
    
    Args:
        dot_path: Path to the .dot file
        png_path: Path to save the .png file
        
    Returns:
        True if successful, False otherwise
    """
    if not HAS_PYGRAPHVIZ:
        return False
    
    try:
        G = pgv.AGraph(dot_path)
        G.layout('dot')
        G.draw(png_path)
        return True
    except Exception as e:
        print(f"Warning: Could not render {dot_path} to PNG: {e}")
        return False


def belief_support_algorithm(env: AtomicPropPOMDP, ltl_formula: str,
                             verbose: bool = False, plot: bool = False,
                             output_dir: str = "figs"):
    """
    Compute a policy for a POMDP with an LTL objective using the
    belief-support algorithm.
    
    This uses the observation-based approach where atomic propositions
    are defined on observations (not states).
    
    Args:
        env: The POMDP environment with atomic propositions on observations
        ltl_formula: The LTL formula to satisfy
        verbose: If True, print detailed information about each step
        plot: If True, save visualizations to files
        output_dir: Directory for saving visualizations
        
    Returns:
        A dict containing the results of the almost-sure winning analysis
    """
    # Create output directory if plotting
    if plot:
        os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1. Translate LTL to a parity automaton
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: Translating LTL formula to parity automaton")
        print("="*70)
        print(f"LTL Formula: {ltl_formula}")
    
    parity_automaton = spot.translate(
        ltl_formula, "parity", "complete", "SBAcc"
    )
    # Splitting edges makes it easier to find the automaton transition
    # for an observation
    parity_automaton = spot.split_edges(parity_automaton)
    
    if verbose:
        print(f"Automaton states: {parity_automaton.num_states()}")
        print(f"Automaton edges: {parity_automaton.num_edges()}")
        print(f"Acceptance condition: {parity_automaton.get_acceptance()}")
        if plot:
            aut_path = os.path.join(output_dir, "automaton.dot")
            with open(aut_path, 'w') as f:
                f.write(parity_automaton.to_str('dot'))
            print(f"Saved automaton to: {aut_path}")

    # STEP 2. Construct the Belief-Support MDP
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: Constructing Belief-Support MDP")
        print("="*70)
        print(f"POMDP states: {len(env.states)}")
        print(f"POMDP actions: {len(env.actions)}")
        print(f"POMDP observations: {len(env.obs)}")
        if plot:
            pomdp_path = os.path.join(output_dir, "pomdp.dot")
            env.show(pomdp_path)
            print(f"Saved POMDP to: {pomdp_path}")
    
    belief_supp_mdp = BeliefSuppMDP(env, parity_automaton)
    
    if verbose:
        print(f"Belief-Support MDP states: {len(belief_supp_mdp.states)}")
        print(f"Belief-Support MDP actions: {len(belief_supp_mdp.actions)}")
        
    # STEP 3. Solve the parity game on the belief-support MDP
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: Solving parity game on Belief-Support MDP")
        print("="*70)
    
    solver = ParityMDPSolver(belief_supp_mdp, verbose=verbose)
    max_prio = max(belief_supp_mdp.prio.values())
    
    if verbose:
        print(f"Maximum priority: {max_prio}")
        
    (aswin, reach_strats, mec_strats) = solver.almostSureWin(
        max_priority=max_prio)
    
    # Extract the POMDP states from the winning belief-support MDP states
    winning_pomdp_states = set()
    for bs_idx in aswin:
        bs = belief_supp_mdp.states[bs_idx]
        # Each belief support is a tuple of (pomdp_state, aut_state) pairs
        if len(bs) == 1:
            winning_pomdp_states.add(bs[0][0])
    
    if verbose:
        print(f"\nAlmost-sure winning states: {len(aswin)} belief-support states")
        print(f"Corresponding to {len(winning_pomdp_states)} POMDP states: "
              f"{sorted(winning_pomdp_states)}")
        
    # Save belief-support MDP visualization if plotting
    if plot:
        bs_mdp_path = os.path.join(output_dir, "belief_support_mdp.dot")
        # mec_strats is already a list of strategy dicts
        belief_supp_mdp.show(bs_mdp_path, reach=aswin,
                             reachStrat=reach_strats,
                             mecs_and_strats=mec_strats)
        if verbose:
            print(f"Saved Belief-Support MDP to: {bs_mdp_path}")
        
        # Try to render all DOT files to PNG
        if HAS_PYGRAPHVIZ:
            if verbose:
                print("\nRendering visualizations to PNG...")
            for name in ["pomdp", "automaton", "belief_support_mdp"]:
                dot_file = os.path.join(output_dir, f"{name}.dot")
                png_file = os.path.join(output_dir, f"{name}.png")
                if os.path.exists(dot_file):
                    if render_dot_to_png(dot_file, png_file):
                        if verbose:
                            print(f"  {name}.png created")
                    else:
                        if verbose:
                            print(f"  {name}.png failed")
        else:
            if verbose:
                print("\nNote: Install pygraphviz to render PNG files")
                print("DOT files can be rendered with: "
                      "dot -Tpng file.dot -o file.png")
    
    return {
        'almost_sure_win_pomdp_states': winning_pomdp_states,
        'almost_sure_win_bs_states': aswin,
        'reachability_strategies': reach_strats,
        'mec_strategies': mec_strats,
        'belief_support_mdp': belief_supp_mdp,
        'parity_automaton': parity_automaton,
        'pomdp': env
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute policies for POMDPs with LTL objectives using "
            "belief-support algorithm."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --file examples/ltl-revealing-tiger.pomdp --ltl_formula "G F p0 & G \! p1"
  %(prog)s --file examples/ltl-corridor-easy.pomdp --ltl_formula "G(p0 & X \! p0 -> X( \! p0 U p1)) & G(p1 & X \!p1 -> X( \!p1 U p0))  & GFp0 & GFp1" --verbose --plot
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the POMDP file to process",
        default="examples/ltl-revealing-tiger-obs.pomdp",
    )
    
    parser.add_argument(
        "--ltl_formula",
        type=str,
        help='LTL formula to satisfy (e.g., "G F p0 & G !p1")',
        default="G F p0 & G !p1",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information about each step",
    )
    
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate visualizations (saved as DOT files in figs/ directory)",
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="figs",
        help="Directory for saving visualizations (default: figs/)",
    )
        
    args = parser.parse_args()
    
    # Load the POMDP file
    try:
        with open(args.file, 'r') as f:
            content = f.read()
        pomdp = pomdp_parser.parse(content)
        print(f"Loaded POMDP from: {args.file}")
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return 1
    except Exception as e:
        print(f"Error parsing POMDP file: {e}")
        return 1
    
    # Check that we have the right type for this algorithm
    if not isinstance(pomdp, AtomicPropPOMDP):
        print(f"Error: belief_support_algorithm requires a POMDP with "
              f"atomic propositions (AtomicPropPOMDP), "
              f"but got {type(pomdp).__name__}")
        return 1
    
    if not args.verbose:
        print(f"\nLTL Formula: {args.ltl_formula}")
    
    result = belief_support_algorithm(
        pomdp, args.ltl_formula,
        verbose=args.verbose,
        plot=args.plot,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("COMPUTATION COMPLETE")
    print("="*70)
    pomdp_states = sorted(result['almost_sure_win_pomdp_states'])
    bs_states = result['almost_sure_win_bs_states']
    print(f"  Almost-sure winning POMDP states: {pomdp_states}\n  Listed by name: {', '.join(map(lambda s: pomdp.states[s], pomdp_states))}")
    print(f"  (from {len(bs_states)} belief-support states)")
    
    if args.plot:
        print(f"\nVisualizations saved to: {args.output_dir}/")
        if HAS_PYGRAPHVIZ:
            print("  - pomdp.png (POMDP structure)")
            print("  - automaton.png (Parity automaton)")
            print("  - belief_support_mdp.png "
                  "(Belief-support MDP with strategy)")
        else:
            print("  - pomdp.dot (POMDP structure)")
            print("  - automaton.dot (Parity automaton)")
            print("  - belief_support_mdp.dot "
                  "(Belief-support MDP with strategy)")
            print("\nNote: Install pygraphviz for PNG rendering, or use:")
            print("  dot -Tpng <file>.dot -o <file>.png")
    
    return 0


if __name__ == "__main__":
    exit(main())
