import spot
from pomdpy.parsers import pomdp
from pomdpy.pomdp import POMDP
from itertools import product as setproduct

# we may use spot.split_edges at some point


def get_product_state_index(env, s, a):
    return a * len(env.states) + s


def get_next_state(env, aut, s, a):
    bdict = aut.get_dict()
    formula = env.generateFormula(s)
    transitions = aut.out(a)
    for t in transitions:
        if formula == spot.bdd_format_formula(bdict, t.cond):
            return t.dst
    assert False


def init_product(env, aut):
    product = POMDP()
    aut = spot.split_edges(aut)
    product_states = [
        f"{s}-{i}" for s, i in setproduct(env.states, range(aut.num_states()))
    ]
    product.setStates(len(product_states))
    product.setActions(env.actions)
    product.setObs(env.obs)
    return product


def get_state_name(product, env, aut, idx):
    env_idx = idx // len(env.states)
    aut_idx = idx % aut.num_states()
    return f"{env.states[env_idx]}-{aut_idx}"


def set_start_probs(product: POMDP, env: POMDP, aut):
    for start_state in env.start:
        a_ = get_next_state(env, aut, start_state, aut.get_init_state_number())
        state = get_product_state_index(env, start_state, a_)
        product.start[state] = env.start[start_state]


def set_transition_probs(product: POMDP, env: POMDP, aut):
    for a in range(aut.num_states()):
        for src in env.trans:
            src_product = get_product_state_index(env, src, a)
            for act in env.trans[src]:
                for dst in env.trans[src][act]:
                    a_ = get_next_state(env, aut, dst, a)
                    dst_product = get_product_state_index(env, dst, a_)
                    p = env.trans[src][act][dst]
                    product._addOneTrans(src_product, act, dst_product, p)


def set_obs_probs(product: POMDP, env: POMDP, aut):
    # action -> dst -> obs
    for a in range(aut.num_states()):
        for act in env.obsfun:
            for dst in env.obsfun[act]:
                for obs in env.obsfun[act][dst]:
                    p = env.obsfun[act][dst][obs]
                    product._addOneObs(
                        act, get_product_state_index(env, dst, a), obs, p
                    )


def set_priorities(product: POMDP, env: POMDP, aut):
    simplified_buchi = aut.acc().num_sets() == 1
    for a in range(aut.num_states()):
        for s in range(len(env.states)):
            for prio in list(aut.state_acc_sets(a).sets()):
                product.addPriority(
                    prio + (2 if simplified_buchi else 0),
                    [get_product_state_index(env, s, a)],
                    ids=True,
                )
            if list(aut.state_acc_sets(a).sets()) == []:
                product.addPriority(1, [get_product_state_index(env, s, a)], ids=True)
