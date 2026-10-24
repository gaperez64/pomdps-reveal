import argparse
import os
import re
import signal
from pomdpy.pomdp import AtomicPropPOMDP, ParityPOMDP
from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.parsers.tlsf import parse_tlsf_file
from pomdpy.product import ProductPOMDP
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
from pomdpy.revealing import is_strongly_revealing
import spot


class TimeoutError(Exception):
    """Raised when computation exceeds timeout"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Computation exceeded timeout")


try:
    import pygraphviz as pgv
    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False


def extract_atoms_from_formula(ltl_formula):
    """Extract which atomic propositions (p0, p1, p2, ...) are used in LTL formula."""
    matches = re.findall(r'p(\d+)', ltl_formula)
    return sorted(set(int(m) for m in matches))


def get_atoms_from_pomdp_and_tlsf(pomdp, tlsf_data, ltl_formula):
    """
    Determine which atomic propositions to use based on POMDP, TLSF, and formula.
    
    Strategy:
    - Use all atoms defined in the POMDP
    - Warn if TLSF or formula mentions atoms not defined in the POMDP
    
    Returns:
        list of atom indices to use (all atoms from POMDP)
    """
    pomdp_atoms = sorted(pomdp.atoms.keys())
    
    # Determine what TLSF or formula expects
    if tlsf_data and tlsf_data.get('inputs'):
        spec_atoms = tlsf_data['inputs']
        source = "TLSF"
    else:
        spec_atoms = extract_atoms_from_formula(ltl_formula)
        source = "formula"
    
    missing_atoms = set(spec_atoms) - set(pomdp_atoms)
    if missing_atoms:
        print(f"\nWarning: {source} mentions atoms not defined in the POMDP: {sorted(missing_atoms)}")
    
    return pomdp_atoms


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


def belief_support_algorithm(env, ltl_formula: str = None,
                             verbose: bool = False, plot: bool = False,
                             output_dir: str = "figs",
                             atoms: list = None,
                             parity_automaton=None):
    """
    Compute a policy for a POMDP with an LTL objective using the
    belief-support algorithm.
    
    Supports two types of input:
    1. AtomicPropPOMDP + LTL formula: Constructs parity automaton from LTL (steps 1-4)
    2. ParityPOMDP: Uses existing priorities directly (skips to step 3)

    Inputs must be strongly revealing; for LTL, revealing alternatives must
    also preserve labels.
    
    Args:
        env: Either AtomicPropPOMDP (with atomic propositions on observations)
             or ParityPOMDP (with priorities already defined)
        ltl_formula: The LTL formula to satisfy (required for AtomicPropPOMDP, ignored for ParityPOMDP)
        verbose: If True, print detailed information about each step
        plot: If True, save visualizations to files
        output_dir: Directory for saving visualizations
        atoms: List of atomic proposition indices to use (e.g., [0, 1]).
               Only used for AtomicPropPOMDP. If None, will be extracted from ltl_formula.
        parity_automaton: Pre-constructed parity automaton (optional, for AtomicPropPOMDP)
        
    Returns:
        A dict containing the results of the almost-sure winning analysis.
        initial_winning refers to the given initial belief.
        almost_sure_win_pomdp_states projects reachable winning singleton
        beliefs onto POMDP states; for LTL, winning may depend on the omitted
        automaton state and need not hold when starting afresh there.
        visualization_files lists the DOT and PNG files written by this run.
    """
    # Determine input type and workflow
    is_parity_pomdp = isinstance(env, ParityPOMDP)
    is_atomic_prop_pomdp = isinstance(env, AtomicPropPOMDP)
    
    if not is_parity_pomdp and not is_atomic_prop_pomdp:
        raise ValueError(f"Expected ParityPOMDP or AtomicPropPOMDP, got {type(env).__name__}")
    
    # Create output directory if plotting
    visualization_files = []
    if plot:
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Input type: {type(env).__name__}")
        print(f"POMDP states: {len(env.states)}")
        print(f"POMDP actions: {len(env.actions)}")
        print(f"POMDP observations: {len(env.obs)}")
        if is_atomic_prop_pomdp:
            print(f"Atomic propositions: {sorted(env.atoms.keys())}")
        elif is_parity_pomdp:
            print(f"Priorities: {sorted(env.prio.keys())}")
    
    # For AtomicPropPOMDP, validate or extract atomic propositions
    if is_atomic_prop_pomdp:
        if ltl_formula is None:
            raise ValueError("ltl_formula is required for AtomicPropPOMDP")
            
        if atoms is None:
            atoms = extract_atoms_from_formula(ltl_formula)
            if verbose:
                print(f"Atomic propositions extracted from formula: {atoms}")
        missing_atoms = (set(atoms) | set(extract_atoms_from_formula(ltl_formula))) - set(env.atoms)
        if missing_atoms:
            print(f"Warning: Atoms not defined in the POMDP: {sorted(missing_atoms)}")
    
    # For AtomicPropPOMDP: Steps 1-2 (construct product POMDP)
    if is_atomic_prop_pomdp:
        # The product matches observations using only the automaton's atoms.
        # STEP 1. Translate LTL to a parity automaton
        if parity_automaton is None:
            if verbose:
                print("\n" + "="*70)
                print("STEP 1: Translating LTL formula to parity automaton")
                print("="*70)
                print(f"LTL Formula: {ltl_formula}")
            
            # Register the selected propositions in the shared BDD dictionary.
            env_dict = spot.make_bdd_dict()
            try:
                for atom_id in atoms:
                    env_dict.register_proposition(f"p{atom_id}", env_dict)

                parity_automaton = spot.translate(
                    ltl_formula, "parity max even", "deterministic", "complete",
                    "SBAcc", "colored", dict=env_dict
                )
            finally:
                # The automaton keeps its own registrations.
                env_dict.unregister_all_my_variables(env_dict)
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
                visualization_files.append(aut_path)
                print(f"Saved automaton to: {aut_path}")

        # STEP 2. Construct the product POMDP
        if verbose:
            print("\n" + "="*70)
            print("STEP 2: Constructing a ParityPOMDP as product of an AtomicPropPOMDP and a parity automaton")
            print("="*70)
            
        product_pomdp = ProductPOMDP(env, parity_automaton)

        if verbose:
            print(f"Product POMDP states: {len(product_pomdp.states)}")
            print(f"Product POMDP actions: {len(product_pomdp.actions)}")
            print(f"Product POMDP observations: {len(product_pomdp.obs)}")
        if plot:
            product_path = os.path.join(output_dir, "product_pomdp.dot")
            with open(product_path, 'w') as f:
                f.write(product_pomdp.to_str('dot'))
            visualization_files.append(product_path)
            print(f"Saved product POMDP to: {product_path}")
    else:
        # For ParityPOMDP: Skip steps 1-2, use directly
        if verbose:
            print("\n" + "="*70)
            print("STEPS 1-2 SKIPPED: Input is already a ParityPOMDP")
            print("="*70)
        product_pomdp = env
        # No parity automaton in this case (priorities are on states)

    # STEP 3. Construct the Belief-Support MDP from the ParityPOMDP
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: Constructing Belief-Support MDP from ParityPOMDP")
        print("="*70)
    
    # BeliefSuppMDP constructor signature: (parity_pomdp, parity_automaton=None)
    # For ParityPOMDP input, parity_automaton should be None
    belief_supp_mdp = BeliefSuppMDP(product_pomdp, parity_automaton if is_atomic_prop_pomdp else None)

    if verbose:
        print(f"Belief-Support MDP states: {len(belief_supp_mdp.states)}")
        print(f"Belief-Support MDP actions: {len(belief_supp_mdp.actions)}")

    # STEP 4. Solve the parity game on the belief-support MDP
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: Solving parity game on Belief-Support MDP")
        print("="*70)
    
    solver = ParityMDPSolver(belief_supp_mdp, verbose=verbose)
    max_prio = max(belief_supp_mdp.prio.values())
    
    if verbose:
        print(f"Maximum priority: {max_prio}")
        
    (aswin, reach_strats, mec_strats) = solver.almostSureWin(
        max_priority=max_prio)
    
    # Project winning singleton beliefs onto POMDP states.
    # For LTL, the omitted automaton state records the observation history.
    winning_pomdp_states = set()
    for bs_idx in aswin:
        bs = belief_supp_mdp.states[bs_idx]
        if len(bs) == 1:
            if is_atomic_prop_pomdp:
                # Product beliefs already store (POMDP state, automaton state).
                state_idx, _ = bs[0]
            else:
                # For ParityPOMDP: product_pomdp == env, so state is direct
                state_idx = bs[0]
            winning_pomdp_states.add(state_idx)
    
    if verbose:
        print(f"\nAlmost-sure winning states: {len(aswin)} belief-support states")
        print(f"POMDP states occurring in winning singleton beliefs: "
              f"{sorted(winning_pomdp_states)}")
        
    # Save belief-support MDP visualization if plotting
    if plot:
        bs_mdp_path = os.path.join(output_dir, "belief_support_mdp.dot")
        # mec_strats is already a list of strategy dicts
        belief_supp_mdp.show(bs_mdp_path, reach=aswin,
                             reachStrat=reach_strats,
                             mecs_and_strats=mec_strats)
        visualization_files.append(bs_mdp_path)
        if verbose:
            print(f"Saved Belief-Support MDP to: {bs_mdp_path}")
        
        # Render the automaton and belief-support plots written by this run.
        if HAS_PYGRAPHVIZ:
            if verbose:
                print("\nRendering visualizations to PNG...")
            for name in ["automaton", "belief_support_mdp"]:
                dot_file = os.path.join(output_dir, f"{name}.dot")
                png_file = os.path.join(output_dir, f"{name}.png")
                if dot_file in visualization_files:
                    if render_dot_to_png(dot_file, png_file):
                        visualization_files.append(png_file)
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
        'initial_winning': belief_supp_mdp.start in aswin,
        'almost_sure_win_pomdp_states': winning_pomdp_states,
        'almost_sure_win_bs_states': aswin,
        'reachability_strategies': reach_strats,
        'mec_strategies': mec_strats,
        'belief_support_mdp': belief_supp_mdp,
        'parity_automaton': parity_automaton if is_atomic_prop_pomdp else None,
        'pomdp': env,
        'product_pomdp': product_pomdp,
        'visualization_files': visualization_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute policies for POMDPs with LTL objectives using "
            "belief-support algorithm."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using TLSF file (recommended - auto-detects atoms and formula)
  %(prog)s --file examples/ltl-revealing/ltl-tiger-repeating.pomdp --tlsf_file examples/ltl-revealing/ltl-tiger-repeating.tlsf
  
  # TLSF with verbose output
  %(prog)s --file examples/ltl-revealing/ltl-corridor-easy.pomdp --tlsf_file examples/ltl-revealing/ltl-corridor-easy.tlsf --verbose --plot
  
  # Using LTL formula with explicit atoms
  %(prog)s --file examples/ltl-revealing/ltl-tiger-repeating.pomdp --ltl_formula "F p0 & G !p1" --atoms "0,1"
  
  # Using LTL formula with auto-detection (atoms taken from the POMDP)
  %(prog)s --file examples/ltl-revealing/ltl-tiger-repeating.pomdp --ltl_formula "F p0 & G !p1"
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the POMDP file to process",
        default="examples/ltl-revealing/ltl-tiger-repeating.pomdp",
    )
    
    parser.add_argument(
        "--ltl_formula",
        type=str,
        help='LTL formula to satisfy (e.g., "G F p0 & G ! p1"). If --tlsf_file is provided, the TLSF specification takes precedence.',
        default=None,
    )
    
    parser.add_argument(
        "--tlsf_file",
        type=str,
        help='Path to TLSF specification file, converted to LTL using tlsf-tools.',
        default=None,
    )
    
    parser.add_argument(
        "--atoms",
        type=str,
        help=(
            'Comma-separated list of atomic proposition indices to use '
            '(e.g., "0,1" for p0 and p1). If not specified, will be '
            'taken from the POMDP.'
        ),
        default=None,
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
    
    # Check that we have a supported type for this algorithm
    if not isinstance(pomdp, (AtomicPropPOMDP, ParityPOMDP)):
        print(f"Error: belief_support_algorithm requires AtomicPropPOMDP or ParityPOMDP, "
              f"but got {type(pomdp).__name__}")
        return 1
    
    is_parity_pomdp = isinstance(pomdp, ParityPOMDP)
    is_atomic_prop_pomdp = isinstance(pomdp, AtomicPropPOMDP)
    
    # LTL inputs also need revealing alternatives with the same labels.
    condition = "strongly revealing"
    if is_atomic_prop_pomdp:
        condition += " with label-preserving alternatives"
    print(f"Checking if POMDP is {condition}...")
    if is_strongly_revealing(pomdp, preserve_labels=is_atomic_prop_pomdp):
        print(f"✓ POMDP is {condition}")
    else:
        print(f"✗ POMDP is NOT {condition}")
        print("Warning: The belief-support algorithm assumes this condition.")
        print("         Results may not be correct when it fails.")
    print()
    
    # Parse TLSF file if provided
    tlsf_data = None
    if args.tlsf_file and is_atomic_prop_pomdp:
        try:
            tlsf_data = parse_tlsf_file(args.tlsf_file)
            print(f"Loaded TLSF from: {args.tlsf_file}")
        except FileNotFoundError:
            print(f"Error: TLSF file '{args.tlsf_file}' not found.")
            return 1
        except Exception as e:
            print(f"Error parsing TLSF file: {e}")
            return 1
    
    # Determine LTL formula to use (required for AtomicPropPOMDP, not for ParityPOMDP)
    ltl_formula = None
    if is_atomic_prop_pomdp:
        if tlsf_data is not None:
            ltl_formula = tlsf_data['formula']
        elif args.ltl_formula:
            ltl_formula = args.ltl_formula
        else:
            print("Error: AtomicPropPOMDP requires LTL formula. Use --ltl_formula or --tlsf_file.")
            return 1
    elif is_parity_pomdp:
        # ParityPOMDP doesn't need LTL formula (priorities are already defined)
        if args.ltl_formula or args.tlsf_file:
            print("Note: ParityPOMDP input detected. LTL formula/TLSF will be ignored.")
        print("Using priorities from ParityPOMDP directly (skipping LTL translation).")
    
    if ltl_formula and not args.verbose:
        print(f"\nLTL Formula: {ltl_formula}")
    
    # Determine atoms to use (only for AtomicPropPOMDP)
    atoms = None
    if is_atomic_prop_pomdp:
        if args.atoms:
            # Explicitly specified atoms take priority
            try:
                atoms = [int(x.strip()) for x in args.atoms.split(',')]
                if not args.verbose:
                    print(f"Using atomic propositions (explicit): {atoms}")
            except ValueError:
                print(f"Error: Invalid atoms specification '{args.atoms}'. "
                      f"Expected comma-separated integers.")
                return 1
        else:
            # Auto-determine atoms from POMDP, TLSF, and formula
            atoms = get_atoms_from_pomdp_and_tlsf(pomdp, tlsf_data, ltl_formula)
            if not args.verbose:
                print(f"Using atomic propositions (auto-detected): {atoms}")
            
    # Set up timeout (30 seconds)
    timeout_seconds = 30
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = belief_support_algorithm(
            pomdp, ltl_formula,
            verbose=args.verbose,
            plot=args.plot,
            output_dir=args.output_dir,
            atoms=atoms
        )
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        print(f"\n{'='*70}")
        print(f"TIMEOUT: Computation exceeded {timeout_seconds} seconds")
        print(f"{'='*70}")
        return 2
    finally:
        signal.alarm(0)  # Cancel the alarm in case of other errors
    
    print("\n" + "="*70)
    print("COMPUTATION COMPLETE")
    print("="*70)
    pomdp_states = sorted(result['almost_sure_win_pomdp_states'])
    bs_states = result['almost_sure_win_bs_states']
    print(f"  Initial belief is almost-sure winning: {'yes' if result['initial_winning'] else 'no'}")
    print(f"  Almost-sure winning belief-support states: {len(bs_states)}")
    print(f"  POMDP states occurring in winning singleton beliefs: {pomdp_states}")
    print(f"  Listed by name: {', '.join(pomdp.states[s] for s in pomdp_states)}")
    if is_atomic_prop_pomdp:
        print("  For LTL, winning here can depend on past observations.")
    
    if args.plot:
        print(f"\nVisualizations saved to: {args.output_dir}/")
        for path in result['visualization_files']:
            print(f"  - {os.path.basename(path)}")
        if not HAS_PYGRAPHVIZ:
            print("\nNote: Install pygraphviz for PNG rendering, or use:")
            print("  dot -Tpng <file>.dot -o <file>.png")
    
    return 0


if __name__ == "__main__":
    exit(main())
