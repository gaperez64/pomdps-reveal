import spot
from pomdpy.pomdp import AtomicPropPOMDP, ParityPOMDP
from itertools import product as setproduct
from typing import Any

spot_automaton = Any

def get_automaton_atoms(aut: spot_automaton):
    """Extract which atomic propositions the automaton uses."""
    import re
    hoa = aut.to_str('hoa')
    for line in hoa.split('\n'):
        if line.startswith('AP:'):
            # Parse: "AP: 2 "p0" "p1"" -> [0, 1]
            parts = line.split('"')
            aps = [parts[i] for i in range(1, len(parts), 2)]
            atom_ids = [int(re.search(r'\d+', ap).group()) for ap in aps]
            return sorted(atom_ids)
    return []


def get_product_state_index(env: AtomicPropPOMDP, state_pomdp_idx: int, state_aut_idx: int):
    return state_aut_idx * len(env.states) + state_pomdp_idx


def get_next_aut_state_by_observation(env: AtomicPropPOMDP,
                                       aut: spot_automaton,
                                       obs_idx: int,
                                       state_aut_idx: int,
                                       aut_atoms=None):
    """
    Find the next automaton state for a given observation.
    
    Args:
        env: The POMDP environment
        aut: The automaton
        obs_idx: Observation index
        state_aut_idx: Current automaton state
        aut_atoms: List of atom IDs used by the automaton.
                  If None, will be extracted from automaton.
    """
    if aut_atoms is None:
        aut_atoms = get_automaton_atoms(aut)
    
    bdict = aut.get_dict()
    formula = env.generate_observation_formula(obs_idx, atoms_to_use=aut_atoms)
    for t in aut.out(state_aut_idx):
        if formula == spot.bdd_format_formula(bdict, t.cond):
            return t.dst
    raise ValueError(
        f"No matching transition found for observation {obs_idx} "
        f"(formula: {formula}) in automaton state {state_aut_idx}"
    )


def init_product(env: AtomicPropPOMDP, aut: spot_automaton):
    product = ParityPOMDP()
    product_states = [
        f"{s}-{i}"
        for i, s in setproduct(
            range(aut.num_states()),
            env.states,
        )
    ]
    product.setStates(product_states)
    product.setActions(env.actions)
    product.setObs(env.obs)
    return product


def get_state_name(product: ParityPOMDP, env: AtomicPropPOMDP, aut: spot_automaton, idx: int):
    env_idx = idx % len(env.states)
    aut_idx = idx // len(env.states)
    return f"{env.states[env_idx]}-{aut_idx}"


def set_start_probs(product: ParityPOMDP, env: AtomicPropPOMDP, aut: spot_automaton):
    # Start in the initial automaton state for all starting POMDP states
    aut_init = aut.get_init_state_number()
    for s_idx, prob in env.start.items():
        state = get_product_state_index(env, s_idx, aut_init)
        product.start[state] = prob


def build_transitions(product: ParityPOMDP, env: AtomicPropPOMDP, aut: spot_automaton):
    # Directly populate product.transitions over (next_state, obs)
    # for each product state and action using env.T and env.O
    for q in range(aut.num_states()):
        for s in env.T:
            src_idx = get_product_state_index(env, s, q)
            if src_idx not in product.transitions:
                product.transitions[src_idx] = {}
            for a in env.T[s]:
                if a not in product.transitions[src_idx]:
                    product.transitions[src_idx][a] = {}
                for s_prime, p_t in env.T[s][a].items():
                    # For each possible observation upon arriving at s_prime
                    if a in env.O and s_prime in env.O[a]:
                        for o, p_o in env.O[a][s_prime].items():
                            q_prime = get_next_aut_state_by_observation(env, aut, o, q)
                            dst_idx = get_product_state_index(env, s_prime, q_prime)
                            product.transitions[src_idx][a][(dst_idx, o)] = (
                                product.transitions[src_idx][a].get((dst_idx, o), 0.0)
                                + p_t * p_o
                            )


def set_priorities(product: ParityPOMDP, env: AtomicPropPOMDP, aut: spot_automaton):
    simplified_buchi = aut.acc().num_sets() == 1
    for state_aut_idx in range(aut.num_states()):
        for state_pomdp_idx in range(len(env.states)):
            product_state_index = get_product_state_index(env, state_pomdp_idx, state_aut_idx)
            if list(aut.state_acc_sets(state_aut_idx).sets()) == []:
                product.addPriority(1, [get_product_state_index(env, state_pomdp_idx, state_aut_idx)], ids=True)
            else:                
                for prio in list(aut.state_acc_sets(state_aut_idx).sets()):
                    product.addPriority(
                        prio + (2 if simplified_buchi else 0),
                        [product_state_index],
                        ids=True,
                    )

    product.prio = dict(sorted(product.prio.items()))


class ProductPOMDP(ParityPOMDP):
    """Product between an AtomicPropPOMDP and a parity automaton.

    - States are pairs (s, q) flattened to index q*|S| + s and named "state-q".
    - Actions are inherited from the POMDP.
    - Observations are inherited from the POMDP.
    - Transitions combine POMDP dynamics with automaton transitions based on
      the observation-based atomic propositions.
    - Priorities come from the automaton state acceptance sets.
    """

    def __init__(self, env: AtomicPropPOMDP, aut: spot_automaton):
        super().__init__()
        self._env = env
        self._aut = aut

        # Initialize base structure from components
        prod = init_product(env, aut)
        self.setStates(prod.states)
        self.statesinv = prod.statesinv
        self.setActions(prod.actions)
        self.actionsinv = prod.actionsinv
        self.setObs(prod.obs)
        self.obsinv = prod.obsinv

        # Start distribution: from env.start and initial automaton state
        set_start_probs(self, env, aut)

        # Build full transitions directly combining env.T and env.O
        build_transitions(self, env, aut)

        # Priorities on product states from automaton acceptance
        set_priorities(self, env, aut)

    def state_name(self, idx: int):
        return get_state_name(self, self._env, self._aut, idx)
