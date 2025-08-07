import spot
from pomdpy.pomdp import POMDP
from itertools import product as setproduct
from typing import Any

# we may use spot.split_edges at some point

spot_automaton = Any

def get_product_state_index(env: POMDP, state_pomdp_idx: int, state_aut_idx: int):
    return state_aut_idx * len(env.states) + state_pomdp_idx


def get_next_state(env: POMDP, aut: spot_automaton, state_pomdp_idx: int, state_aut_idx: int):
    bdict = aut.get_dict()
    formula = env.generateFormula(state_pomdp_idx)
    transitions = aut.out(state_aut_idx)
    for t in transitions:
        if formula == spot.bdd_format_formula(bdict, t.cond):
            return t.dst
    raise ValueError(f"No matching transition found for formula: {formula} in automaton state {state_aut_idx}")


def init_product(env: POMDP, aut: spot_automaton):
    product = POMDP()
    aut = spot.split_edges(aut)
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


def get_state_name(product: POMDP, env: POMDP, aut: spot_automaton, idx: int):
    env_idx = idx % len(env.states)
    aut_idx = idx // len(env.states)
    return f"{env.states[env_idx]}-{aut_idx}"


def set_start_probs(product: POMDP, env: POMDP, aut: spot_automaton):
    for start_state in env.start:
        next_aut_idx = get_next_state(env, aut, start_state, aut.get_init_state_number())
        state = get_product_state_index(env, start_state, next_aut_idx)
        product.start[state] = env.start[start_state]


def set_transition_probs(product: POMDP, env: POMDP, aut: spot_automaton):
    for state_aut_idx in range(aut.num_states()):
        for src in env.trans:
            src_product = get_product_state_index(env, src, state_aut_idx)
            for act in env.trans[src]:
                for dst in env.trans[src][act]:
                    next_aut_idx = get_next_state(env, aut, dst, state_aut_idx)
                    dst_product = get_product_state_index(env, dst, next_aut_idx)
                    p = env.trans[src][act][dst]
                    product._addOneTrans(src_product, act, dst_product, p)


def set_obs_probs(product: POMDP, env: POMDP, aut: spot_automaton):
    # action -> dst -> obs
    for state_aut_idx in range(aut.num_states()):
        for act in env.obsfun:
            for dst in env.obsfun[act]:
                for obs in env.obsfun[act][dst]:
                    p = env.obsfun[act][dst][obs]
                    product._addOneObs(
                        act, get_product_state_index(env, dst, state_aut_idx), obs, p
                    )


def set_priorities(product: POMDP, env: POMDP, aut: spot_automaton):
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
