from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.pomdp import POMDP
from pomdpy.parsers import pomdp
from pomdpy.product import (
    init_product,
    set_start_probs,
    get_product_state_index,
    get_next_state,
    set_transition_probs,
    set_priorities,
    set_obs_probs
)

import spot


def test_next_state():
    env = POMDP()
    env.setStates(["a", "b"])
    env.prio[0] = set([0])
    env.prio[1] = set([0, 1])
    aut = spot.translate("GFp0 & GFp1", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    assert get_next_state(env, aut, 0, 0) == 0

    env.prio[1] = set([1])
    assert get_next_state(env, aut, 0, 0) == 1

    env.prio[0] = set()
    assert get_next_state(env, aut, 0, 0) == 2


def test_number_of_states():
    env = POMDP()
    env.setStates([str(i) for i in range(10)])
    aut = spot.translate("GFp0 & GFp1", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    assert aut.num_states() == 5

    product = init_product(env, aut)
    assert len(product.states) == len(env.states) * aut.num_states()


def test_ititial_state():
    env = POMDP()
    env.setStates(["a", "b"])
    env.start[env.statesinv["a"]] = 1.0
    env.prio[0] = set([env.statesinv["a"]])

    aut = spot.translate("FGp0", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    product = init_product(env, aut)
    set_start_probs(product, env, aut)
    assert product.start[get_product_state_index(env, 0, 0)] == 1.0

    ####

    env = POMDP()
    env.setStates(["a", "b"])
    env.start[env.statesinv["a"]] = 0.25
    env.start[env.statesinv["b"]] = 0.75
    env.prio[0] = set([0, 1])
    env.prio[1] = set([0])

    aut = spot.translate("GFp0 & GFp1", "parity", "SBAcc")
    aut = spot.split_edges(aut)
    product = init_product(env, aut)
    set_start_probs(product, env, aut)
    assert product.start[get_product_state_index(env, 0, 0)] == 0.25
    assert product.start[get_product_state_index(env, 1, 1)] == 0.75


def test_transitions():
    env = POMDP()
    env.setStates(["a", "b"])
    env.setActions(["act1"])
    env._addOneTrans(0, 0, 0, 0.1)
    env._addOneTrans(0, 0, 1, 0.9)
    env._addOneTrans(1, 0, 1, 0.2)
    env._addOneTrans(1, 0, 0, 0.8)

    env.prio[0] = set([0, 1])
    env.prio[1] = set([0])

    aut = spot.translate("GFp0 & GFp1", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = init_product(env, aut)
    set_transition_probs(product, env, aut)
    print(env.trans)
    print(product.trans)
    assert product.trans[0][0][0] == 0.1
    assert product.trans[0][0][3] == 0.9
    assert product.trans[9][0][0] == 0.8
    assert product.trans[9][0][7] == 0.2


def test_observations():
    env = POMDP()
    env.setStates(["a", "b"])
    env.setActions(["act1"])
    env.setObs(["obs1", "obs2"])
    # obsfun: action -> dst -> obs
    env._addOneObs(0, 0, 0, 0.6)  # act1, a, obs1
    env._addOneObs(0, 0, 1, 0.4)  # act1, a, obs2
    env._addOneObs(0, 1, 0, 0.7)  # act1, b, obs1
    env._addOneObs(0, 1, 1, 0.3)  # act1, b, obs2

    aut = spot.translate("GFp0 & GFp1", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = init_product(env, aut)
    set_obs_probs(product, env, aut)

    for a in range(aut.num_states()):
        idx_a = get_product_state_index(env, 0, a)
        idx_b = get_product_state_index(env, 1, a)
        # act1 = 0, obs1 = 0, obs2 = 1
        assert product.obsfun[0][idx_a][0] == 0.6
        assert product.obsfun[0][idx_a][1] == 0.4
        assert product.obsfun[0][idx_b][0] == 0.7
        assert product.obsfun[0][idx_b][1] == 0.3


def test_priorities():
    env = POMDP()
    env.setStates(["a", "b"])

    aut = spot.translate("GFp0 & GFp1", "parity", "SBAcc")
    aut = spot.split_edges(aut)

    product = init_product(env, aut)
    set_priorities(product, env, aut)

    assert product.prio[2] == set([0, 2, 1, 3])
    assert product.prio[1] == set([4, 6, 8, 5, 7, 9])

