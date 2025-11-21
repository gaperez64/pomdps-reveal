import argparse
import os
import re
import signal
from pomdpy.pomdp import AtomicPropPOMDP
from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.product import ProductPOMDP
from pomdpy.belief_support_MDP import BeliefSuppMDP
from pomdpy.almost_sure_parity_MDP import ParityMDPSolver
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


def parse_tlsf_file(tlsf_path):
    """
    Parse a TLSF file and extract LTL formulas and inputs.
    
    Returns:
        dict with 'formulas' (list of LTL strings) and 'inputs' (list of proposition indices)
    """
    with open(tlsf_path, 'r') as f:
        content = f.read()
    
    result = {'formulas': [], 'inputs': []}
    
    # Extract INPUTS section
    inputs_match = re.search(r'INPUTS\s*\{([^}]+)\}', content, re.DOTALL)
    if inputs_match:
        inputs_text = inputs_match.group(1)
        # Extract p0, p1, p2, etc.
        input_matches = re.findall(r'p(\d+)\s*;', inputs_text)
        result['inputs'] = sorted(set(int(m) for m in input_matches))
    
    # Extract GUARANTEES section
    guarantees_match = re.search(r'GUARANTEES\s*\{([^}]+)\}', content, re.DOTALL)
    if guarantees_match:
        guarantees_text = guarantees_match.group(1)
        # Extract formulas (lines ending with ;, excluding comments)
        lines = guarantees_text.split('\n')
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('//') or not line:
                continue
            # Extract formula (remove trailing semicolon)
            if line.endswith(';'):
                formula = line[:-1].strip()
                # Convert TLSF operators back to standard LTL
                formula = formula.replace('&&', '&').replace('||', '|')
                # Remove backslash escapes (e.g., \! -> !)
                formula = formula.replace('\\!', '!')
                formula = formula.replace('\\&', '&')
                formula = formula.replace('\\|', '|')
                if formula:
                    result['formulas'].append(formula)
    
    return result


def get_atoms_from_pomdp_and_tlsf(pomdp, tlsf_data, ltl_formula):
    """
    Determine which atomic propositions to use based on POMDP, TLSF, and formula.
    
    Strategy:
    - Always use all atoms defined in the POMDP (required for observation formulas)
    - Warn if TLSF or formula uses a different set
    
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
    
    # Always use POMDP atoms (required for observation formula generation)
    if set(spec_atoms) != set(pomdp_atoms):
        print(f"\nNote: {source} uses atoms {spec_atoms}, but POMDP defines {pomdp_atoms}")
        print(f"Using all POMDP atoms {pomdp_atoms} (required for observation formulas)")
    
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


def belief_support_algorithm(env: AtomicPropPOMDP, ltl_formula: str,
                             verbose: bool = False, plot: bool = False,
                             output_dir: str = "figs",
                             atoms: list = None):
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
        atoms: List of atomic proposition indices to use (e.g., [0, 1]).
               If None, will be extracted from ltl_formula with a warning.
        
    Returns:
        A dict containing the results of the almost-sure winning analysis
    """
    # Validate or extract atomic propositions
    if atoms is None:
        atoms = extract_atoms_from_formula(ltl_formula)
        print("\n" + "!"*70)
        print("WARNING: Atomic propositions not specified, extracting from formula")
        print(f"Found atoms in formula: {atoms}")
        print(f"POMDP has atoms: {sorted(env.atoms.keys())}")
        if set(atoms) != set(env.atoms.keys()):
            print("MISMATCH DETECTED: Formula uses different atoms than POMDP defines!")
            print("This may cause errors. Use --atoms to specify correct propositions.")
        print("!"*70 + "\n")
    else:
        # Validate provided atoms
        if set(atoms) != set(env.atoms.keys()):
            print("\n" + "!"*70)
            print("WARNING: Specified atoms don't match POMDP definition")
            print(f"Specified atoms: {atoms}")
            print(f"POMDP has atoms: {sorted(env.atoms.keys())}")
            print("!"*70 + "\n")
    
    # Create output directory if plotting
    if plot:
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"POMDP states: {len(env.states)}")
        print(f"POMDP actions: {len(env.actions)}")
        print(f"POMDP observations: {len(env.obs)}")
        print(f"Atomic propositions: {sorted(env.atoms.keys())}")
    
    # Check if formula uses all atoms - if not, we need to extend it
    # so the automaton alphabet matches the observation formulas
    formula_atoms = extract_atoms_from_formula(ltl_formula)
    all_atoms = sorted(env.atoms.keys()) if atoms is None else atoms
    
    if set(formula_atoms) != set(all_atoms):
        # Formula doesn't mention all atoms - need to add them as free variables
        # We do this by declaring them in spot with spot.default_environment
        if verbose:
            print(f"\nNote: Formula uses atoms {formula_atoms}, but POMDP has {all_atoms}")
            print("Declaring all POMDP atoms in automaton alphabet")
    
    # STEP 1. Translate LTL to a parity automaton
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: Translating LTL formula to parity automaton")
        print("="*70)
        print(f"LTL Formula: {ltl_formula}")
    
    # Declare all atomic propositions to spot before translation
    env_dict = spot.make_bdd_dict()
    for atom_id in all_atoms:
        env_dict.register_proposition(f"p{atom_id}", env_dict)
    
    parity_automaton = spot.translate(
        ltl_formula, "parity", "complete", "SBAcc", dict=env_dict
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
        print(f"Saved product POMDP to: {product_path}")

    # STEP 3. Construct the Belief-Support MDP from the product POMDP
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: Constructing Belief-Support MDP from ParityPOMDP")
        print("="*70)
    
    belief_supp_mdp = BeliefSuppMDP(product_pomdp, parity_automaton)

    if verbose:
        print(f"Belief-Support MDP states: {len(belief_supp_mdp.states)}")
        print(f"Belief-Support MDP actions: {len(belief_supp_mdp.actions)}")

    if plot:
        belief_support_mdp_path = os.path.join(output_dir, "belief_support_mdp.dot")
        with open(belief_support_mdp_path, 'w') as f:
            f.write(env.to_str('dot'))
        print(f"Saved Belief-Support OMDP to: {belief_support_mdp_path}")
        
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
    
    # Extract the POMDP states from the winning belief-support MDP states
    # Keep previous semantics: only extract when belief has a single product state
    winning_pomdp_states = set()
    for bs_idx in aswin:
        bs = belief_supp_mdp.states[bs_idx]
        if len(bs) == 1:
            prod_state_idx, _ = bs[0]
            orig_pomdp_state = prod_state_idx % len(env.states)
            winning_pomdp_states.add(orig_pomdp_state)
    
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
Examples:
  # Using TLSF file (recommended - auto-detects atoms and formula)
  %(prog)s --file examples/ltl/ltl-revealing-tiger.pomdp --tlsf_file examples/ltl/ltl-revealing-tiger.tlsf
  
  # TLSF with verbose output
  %(prog)s --file examples/ltl/ltl-corridor-easy.pomdp --tlsf_file examples/ltl/ltl-corridor-easy.tlsf --verbose --plot
  
  # Using LTL formula with explicit atoms
  %(prog)s --file examples/ltl/ltl-revealing-tiger.pomdp --ltl_formula "F p0 & G !p1" --atoms "0,1"
  
  # Using LTL formula with auto-detection (atoms extracted from formula)
  %(prog)s --file examples/ltl/ltl-revealing-tiger.pomdp --ltl_formula "F p0 & G !p1"
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the POMDP file to process",
        default="examples/ltl-revealing-tiger.pomdp",
    )
    
    parser.add_argument(
        "--ltl_formula",
        type=str,
        help='LTL formula to satisfy (e.g., "G F p0 & G ! p1"). If --tlsf_file is provided, formulas from TLSF take precedence.',
        default=None,
    )
    
    parser.add_argument(
        "--tlsf_file",
        type=str,
        help='Path to TLSF specification file. If provided, LTL formulas and atomic propositions will be extracted from it.',
        default=None,
    )
    
    parser.add_argument(
        "--atoms",
        type=str,
        help=(
            'Comma-separated list of atomic proposition indices to use '
            '(e.g., "0,1" for p0 and p1). If not specified, will be '
            'extracted from the LTL formula with a warning about potential mismatches.'
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
    
    # Check that we have the right type for this algorithm
    if not isinstance(pomdp, AtomicPropPOMDP):
        print(f"Error: belief_support_algorithm requires a POMDP with "
              f"atomic propositions (AtomicPropPOMDP), "
              f"but got {type(pomdp).__name__}")
        return 1
    
    # Parse TLSF file if provided
    tlsf_data = None
    if args.tlsf_file:
        try:
            tlsf_data = parse_tlsf_file(args.tlsf_file)
            print(f"Loaded TLSF from: {args.tlsf_file}")
            if tlsf_data['formulas']:
                print(f"Found {len(tlsf_data['formulas'])} LTL formula(s)")
        except FileNotFoundError:
            print(f"Error: TLSF file '{args.tlsf_file}' not found.")
            return 1
        except Exception as e:
            print(f"Error parsing TLSF file: {e}")
            return 1
    
    # Determine LTL formula to use
    if tlsf_data and tlsf_data['formulas']:
        # Use first formula from TLSF (or combine if multiple)
        if len(tlsf_data['formulas']) == 1:
            ltl_formula = tlsf_data['formulas'][0]
        else:
            # Combine multiple formulas with &
            ltl_formula = ' & '.join(f'({f})' for f in tlsf_data['formulas'])
            print(f"Combined {len(tlsf_data['formulas'])} formulas from TLSF")
    elif args.ltl_formula:
        ltl_formula = args.ltl_formula
    else:
        print("Error: No LTL formula specified. Use --ltl_formula or --tlsf_file.")
        return 1
    
    if not args.verbose:
        print(f"\nLTL Formula: {ltl_formula}")
    
    # Determine atoms to use
    atoms = None
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
        
        # Check if formula uses fewer atoms than POMDP defines
        formula_atoms = extract_atoms_from_formula(ltl_formula)
        if set(formula_atoms) != set(atoms):
            # Extend formula to constrain unused atoms
            # This ensures automaton alphabet matches observation formulas
            unused_atoms = set(atoms) - set(formula_atoms)
            if unused_atoms:
                print(f"Note: Formula doesn't constrain atoms {sorted(unused_atoms)}")
                print(f"Automaton will accept any value for these atoms")
    
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
