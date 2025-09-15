import pygraphviz as pgv
from collections import deque
from pomdpy.pomdp import POMDP
from typing import Any
import copy
import spot


# This helper function is moved from product.py
def get_next_aut_state_by_observation(
    env: POMDP, aut: Any, obs_idx: int, state_aut_idx: int
):
    bdict = aut.get_dict()
    formula = env.generate_observation_formula(obs_idx)
    transitions = aut.out(state_aut_idx)
    for t in transitions:
        if formula == spot.bdd_format_formula(bdict, t.cond):
            return t.dst
    raise ValueError(
        f"No matching transition found for formula: {formula} in automaton state {state_aut_idx} for observation {obs_idx}"
    )


class BeliefSuppAut:

    def __init__(self, env: POMDP, aut: Any):
        """
        Constructs the Belief-Support Automaton directly from a POMDP and a parity automaton.
        """
        self.pomdp = env
        self.aut = aut
        self.actions = env.actions
        self.actionsinv = env.actionsinv
        self.trans = {}  # state -> action -> list of successor belief supports
        self.prio = {}

        # 1. Define the initial belief support.
        # A belief support is a set of (pomdp_state_idx, aut_state_idx) tuples.
        aut_init_idx = self.aut.get_init_state_number()
        initial_pomdp_states = {k for k, v in self.pomdp.start.items() if v > 0}

        st = tuple(sorted([(s, aut_init_idx) for s in initial_pomdp_states]))

        self.states = [st]  # List of belief supports (tuples of product-state tuples)
        self.statesinv = {st: 0}  # Maps belief support to its index
        self.start = 0

        explore = deque([st])

        # 2. Perform forward exploration to build the belief-support automaton graph.
        while len(explore) > 0:
            current_bs = explore.popleft()
            current_bs_idx = self.statesinv[current_bs]
            self.trans[current_bs_idx] = {}
            # print(f"Looking at {current_bs}, idx={current_bs_idx}")

            for act_idx, act in enumerate(self.actions):
                # print(f"\tFor action {act}, idx={act_idx}")
                self.trans[current_bs_idx][act_idx] = []

                # Each possible observation will generate one successor belief support.
                possible_succ_beliefs = {}  # obs_idx -> set of (s', q') tuples

                # Iterate through each product state (s, q) in the current belief support
                for s, q in current_bs:
                    for s_prime, trans_prob in self.pomdp.trans[s][act_idx].items():
                        if trans_prob > 0:
                            if s_prime in self.pomdp.obsfun[act_idx]:
                                for o, obs_prob in self.pomdp.obsfun[act_idx][
                                    s_prime
                                ].items():
                                    if obs_prob > 0:
                                        q_prime = get_next_aut_state_by_observation(
                                            self.pomdp, self.aut, o, q
                                        )
                                        if o not in possible_succ_beliefs:
                                            possible_succ_beliefs[o] = set()
                                        possible_succ_beliefs[o].add((s_prime, q_prime))
                                        # print(f"\t\tFrom {(s, q)} to {(s_prime, q_prime)}, obs={o}")

                possible_succ_beliefs = dict([(key, possible_succ_beliefs[key]) for key in sorted(possible_succ_beliefs)])
                # print(f"\t\t{possible_succ_beliefs}")
                # print()
                # For each observation, create the new belief support and add the transition
                for o, next_bs_set in possible_succ_beliefs.items():
                    next_bs = tuple(sorted(list(next_bs_set)))

                    if not next_bs:
                        continue  # Skip if an observation leads to an empty belief

                    if next_bs in self.statesinv:
                        next_bs_idx = self.statesinv[next_bs]
                    else:
                        next_bs_idx = len(self.states)
                        self.statesinv[next_bs] = next_bs_idx
                        self.states.append(next_bs)
                        explore.append(next_bs)
                        # print(f"\t\tAdding belief {next_bs_idx}: {next_bs}")

                    if next_bs_idx not in self.trans[current_bs_idx][act_idx]:
                        self.trans[current_bs_idx][act_idx].append(next_bs_idx)

                self.trans[current_bs_idx][act_idx] = sorted(self.trans[current_bs_idx][act_idx])

        # 3. Set priorities for the belief supports
        self.setPriorities()

    def setPriorities(self):
        """
        Sets the priority for each belief support. The priority of a belief support
        is the maximum priority of any automaton state within that support.
        """
        simplified_buchi = self.aut.acc().num_sets() == 1

        for i, bs in enumerate(self.states):
            max_prio = 0
            if not bs:  # Handle empty belief supports if they occur
                self.prio[i] = 0
                continue

            for _, q in bs:
                acc_sets = list(self.aut.state_acc_sets(q).sets())
                prio = 0
                if not acc_sets:
                    prio = 0 + (1 if simplified_buchi else 0)
                else:
                    # spot priorities are 0-indexed, so a set {0} has prio 0.
                    # We map to the standard parity game convention.
                    prio = max(acc_sets) + (2 if simplified_buchi else 0)

                if prio > max_prio:
                    max_prio = prio
            self.prio[i] = max_prio
            # print(f"Setting {i} priority = {max_prio}")

    def prettyName(self, st):
        return tuple([self.pomdp.states[q] for q in st])

    def resetPreAct(self):
        self.pre = {}
        self.act = {}
        for i, _ in enumerate(self.states):
            self.act[i] = len(self.actions)
            for a, _ in enumerate(self.actions):
                for dst in self.trans[i][a]:
                    preimage = (i, a)
                    if dst in self.pre:
                        self.pre[dst].add(preimage)
                    else:
                        self.pre[dst] = {preimage}

    def tarjanSCCs(self, states, actions):
        assert len(states) > 0
        # this is Wikipedia's version of Tarjan's algorithm
        idx = 0
        S = []
        m = max(states)
        stateIdx = [None for _ in range(m + 1)]
        stateLow = [None for _ in range(m + 1)]
        onStack = [False for _ in range(m + 1)]
        sccDecomp = []

        # recursive helper
        def _strongConnect(q):
            nonlocal idx
            stateIdx[q] = idx
            stateLow[q] = idx
            idx += 1
            S.append(q)
            onStack[q] = True

            # consider successors
            succ = set()
            for a in actions[q]:
                succ.update([dst for dst in self.trans[q][a] if dst in states])
            for dst in succ:
                if stateIdx[dst] is None:
                    _strongConnect(dst)
                    stateLow[q] = min(stateLow[dst], stateLow[q])
                elif onStack[dst]:
                    stateLow[q] = min(stateIdx[dst], stateLow[q])

            # postprocessing roots
            if stateIdx[q] == stateLow[q]:
                scc = []
                while True:
                    w = S.pop()
                    onStack[w] = False
                    scc.append(w)
                    if q == w:
                        break
                sccDecomp.append(scc)

        # start the recursion
        for s in states:
            if stateIdx[s] is None:
                _strongConnect(s)

        return sccDecomp

    def goodMECs(self, priority=0):
        # This is just Algorithm 47 from Baier + Katoen's Principles of Model
        # Checking with minor modifications to account for the priority
        # function making MECs good (only priority 0 present) or great (has at
        # least one state with priority 2)
        print(f"Computing MEC of priority {priority}")

        assert priority % 2 == 0

        self.resetPreAct()
        pre = self.pre
        new_act = {}

        for i, _ in enumerate(self.states):
            new_act[i] = {idx for _, idx in self.actionsinv.items()}

        # get states with priority less or equal to the given prioirity
        newMECs = [set([i for i, _ in enumerate(self.states) if self.prio[i] <= priority])]
        print(f"Initial MEC: {newMECs}")
        # check that there is at least one state of the given priority
        if not any([self.prio[i] == priority for i in newMECs[0]]):
            print()
            if priority >= 2:
                return self.goodMECs(priority=priority - 2)
            else:
                return ([], {})

        MECs = []
        act = copy.deepcopy(new_act)

        while (MECs != newMECs) or (act != new_act):
            print(f"Refinement: \n{MECs=}, \n{newMECs=}, \n{act=}, \n{new_act=}")
            MECs = newMECs
            newMECs = []

            new_act = copy.deepcopy(act)

            for mec in MECs:
                mec = set(mec)
                statesToRemove = set()

                # Split candidate MEC into SCCs and refine based on whether
                # staying in them can be enforced with probability 1
                SCCs = self.tarjanSCCs(mec, act)
                print(f"{SCCs=}")
                for C in SCCs:
                    C = set(C)
                    print(f"Considering SCC: {C}")
                    for s in C:
                        actsToRemove = []
                        for a in act[s]:
                            for dst in self.trans[s][a]:
                                if dst not in C:
                                    actsToRemove.append(a)
                                    print(f"Removing action {a} from state {s} (may get to {dst})")
                                    break
                        act[s] = act[s] - set(actsToRemove)
                        if len(act[s]) == 0:
                            statesToRemove.add(s)

                # Removing states
                while len(statesToRemove) > 0:
                    s = statesToRemove.pop()
                    mec.discard(s)  # may not be there
                    print(f"Removing state {s} completely")
                    if s in pre:
                        for t, a in pre[s]:
                            if t not in C:
                                continue
                            act[t].discard(a)  # maybe not there
                            if len(act[t]) == 0 and t in mec:
                                statesToRemove.add(t)

                # Putting the split MEC candidate back in the list
                for C in SCCs:
                    C = set(C)
                    res = C & mec
                    if len(res) == 0:
                        continue
                    if (priority != 0) and max([self.prio[o] for o in res]) < priority:
                        continue
                    newMECs.append(res)

        print()
        return (MECs, dict(act))

    def mecs(self):
        return [self.goodMECs(), self.goodMECs(priority=2)]

    def cannotReach(self, targets, forbidden: set[tuple[int, int]]):
        """
        Given a set of target states determine which states can reach one
        or more target states. The 'forbidden' set, is a set of
        state-action pairs which cannot be traversed.
        """
        assert self.pre is not None
        visited = set()
        tovisit = set(targets)
        while len(tovisit) > 0:
            q = tovisit.pop()
            if q in self.pre:
                for i, a in self.pre[q]:
                    if (i, a) in forbidden:  # don't go through forbidden node
                        continue
                    if i not in visited:
                        tovisit.add(i)
            visited.add(q)
        return [i for i, _ in enumerate(self.states) if i not in visited]

    def almostSureReach(self, targets):
        # this code is based on Algo. 45 from the
        # Principles of Model Checking by Baier + Katoen
        self.resetPreAct()
        removed = []
        U = set(self.cannotReach(targets, forbidden=set()))
        removed_sa_pairs: set[tuple[int, int]] = set()  # FIX
        counter = 0
        while True:
            print(counter)
            counter += 1
            R = set(U)
            while len(R) > 0:
                u = R.pop()
                # if u in self.pre:
                for t, a in self.pre[u]:
                    if t not in U:
                        # self.act[t] -= 1  # OLD
                        if (t, a) not in removed_sa_pairs:
                            # FIX: do not remove a state-action pair twice
                            self.act[t] -= 1
                            removed_sa_pairs.add((t, a))
                        # if self.act[t] == 0:  # OLD
                        if self.act[t] == 0 and t not in targets:
                            # FIX: do not remove target states
                            R.add(t)
                            U.add(t)
                del self.pre[u]
                removed.append(u)
            U = set(self.cannotReach(targets, forbidden=removed_sa_pairs)) - U
            if len(U) == 0:
                break

        # FIX:
        # Remove all 'removed' actions. If a state 's' is removed in the 'del
        # pre[u]' statement, then not all state-action pairs of 's' are
        # removed from the predecessor dict. This takes care of that.
        # NOTE:
        # This mutates predecessors inside self.pre, in place!, but only
        # because it is a set object in a dictionary
        for successor, predecessors in self.pre.items():
            predecessors.difference_update(removed_sa_pairs)

        return [i for i, _ in enumerate(self.states) if i not in removed]

    def almostSureWin(self, max_priority=2, vis=None):
        print(f"-- Computing maximal end components --")
        mecs_and_strats = [
            self.goodMECs(priority=i) for i in range(0, max_priority + 1, 2)
        ]
        states_and_strats = [
            (set().union(*mecs), strat) for (mecs, strat) in mecs_and_strats
        ]

        u = set().union(*(states for states, _ in states_and_strats))
        r = self.almostSureReach(u)
        # get a reachability strategy
        reachStrat = {}
        for p in r:
            reachStrat[p] = []
            for a, _ in enumerate(self.actions):
                if all([q in r for q in self.trans[p][a]]):
                    reachStrat[p].append(a)
        # clean good and great strategies
        for s, _ in enumerate(self.states):
            for states, strat in states_and_strats:
                if s in strat and s not in states:
                    del strat[s]
                if s in strat and len(strat[s]) == 0:
                    del strat[s]

        # visualize if needed
        if vis is not None:
            self.show(vis, r, reachStrat, mecs_and_strats=mecs_and_strats)

        print(mecs_and_strats)
        return (r, reachStrat, [strat for (_, strat) in mecs_and_strats])

    def setBuchi(self, buchi, cobuchi, areIds=False):
        cobids = cobuchi if areIds else []
        if not areIds:
            for t in cobuchi:
                if t not in self.pomdp.statesinv:
                    print(f"ERROR: Could not find state {t}")
                    exit(1)
                cobids.append(self.pomdp.statesinv[t])
        bids = buchi if areIds else []
        if not areIds:
            for t in buchi:
                if t not in self.pomdp.statesinv:
                    print(f"ERROR: Could not find state {t}")
                    exit(1)
                bids.append(self.pomdp.statesinv[t])
        locprio = {}
        for s, _ in enumerate(self.pomdp.states):
            if s in cobids:
                locprio[s] = 1
            elif s in bids:
                locprio[s] = 2
            else:
                locprio[s] = 0
        for i, o in enumerate(self.states):
            self.prio[i] = max([locprio[s] for s in o])

    def show(self, outfname, reach=None, reachStrat=None, mecs_and_strats=None):
        # This visualization part needs to be adapted for the new mecs_and_strats format
        G = pgv.AGraph(directed=True, strict=False)

        all_mec_beliefs = set()
        if mecs_and_strats:
            for strat_dict in mecs_and_strats:
                all_mec_beliefs.update(strat_dict.keys())

        for i, s in enumerate(self.states):
            name = self.prettyName(s)
            color = "white"
            if reach is not None and i in reach:
                color = "lightblue"
            if all_mec_beliefs and i in all_mec_beliefs:
                color = "palegreen"

            G.add_node(
                i,
                label=f"B{i}: {name}\nprio={self.prio.get(i, 'N/A')}",
                style="filled",
                fillcolor=color,
            )

        for src, _ in enumerate(self.states):
            # Find which strategy applies to this state
            strat_actions = set()
            is_mec_strat = False
            if mecs_and_strats:
                for strat_dict in mecs_and_strats:
                    if src in strat_dict:
                        strat_actions = strat_dict[src]
                        is_mec_strat = True
                        break
            if not strat_actions and reachStrat and src in reachStrat:
                strat_actions = reachStrat[src]

            if src in self.trans:
                for a, act_name in enumerate(self.actions):
                    edge_color = "gray"
                    if a in strat_actions:
                        edge_color = "green" if is_mec_strat else "blue"

                    if a in self.trans[src]:
                        for dst in self.trans[src][a]:
                            G.add_edge(src, dst, color=edge_color, label=f"{act_name}")
        G.layout("dot")
        G.draw(outfname)
