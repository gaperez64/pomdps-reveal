import pygraphviz as pgv


# This is an automaton that captures the belief support
# behavior of a given POMDP
class BeliefSuppAut:
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
                succ.update([dst for dst in self.trans[q][a]
                             if dst in states])
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

        assert priority % 2 == 0

        self.resetPreAct()
        pre = self.pre
        act = {}
        for i, _ in enumerate(self.states):
            act[i] = set([a for a, _ in enumerate(self.actions)])

        # get states with priority less or equal to the given prioirity
        newMECs = [set([i for i, _ in enumerate(self.states) if self.prio[i] <= priority])]
        # check that there is at least one state of the given priority
        if not any([self.prio[i] == priority for i in newMECs[0]]):
            if priority >= 2:
                return self.goodMECs(priority=priority-2)
            else:
                return ([], {})

        MECs = []

        while MECs != newMECs:
            MECs = newMECs
            newMECs = []
            for m in MECs:
                m = set(m)
                statesToRemove = set()

                # Split candidate MEC into SCCs and refine based on whether
                # staying in them can be enforced with probability 1
                SCCs = self.tarjanSCCs(m, act)
                for C in SCCs:
                    C = set(C)
                    for s in C:
                        actsToRemove = []
                        for a in act[s]:
                            for dst in self.trans[s][a]:
                                if dst not in C:
                                    actsToRemove.append(a)
                                    break
                        act[s] = act[s] - set(actsToRemove)
                        if len(act[s]) == 0:
                            statesToRemove.add(s)

                # Removing states
                while len(statesToRemove) > 0:
                    s = statesToRemove.pop()
                    m.discard(s)  # may not be there
                    if s in pre:
                        for (t, a) in pre[s]:
                            if t not in C:
                                continue
                            act[t].discard(a)  # maybe not there
                            if len(act[t]) == 0 and t in m:
                                statesToRemove.add(t)

                # Putting the split MEC candidate back in the list
                for C in SCCs:
                    C = set(C)
                    res = C & m
                    if len(res) == 0:
                        continue
                    if (priority!=0) and max([self.prio[o] for o in res]) < priority:
                        continue
                    newMECs.append(res)
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
        while True:
            R = set(U)
            while len(R) > 0:
                u = R.pop()
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
            for (states, strat) in states_and_strats:
                if s in strat and s not in states:
                    del strat[s]
                if s in strat and len(strat[s]) == 0:
                    del strat[s]

        # visualize if needed
        if vis is not None:
            self.show(vis, r, reachStrat, mecs_and_strats = mecs_and_strats)
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

    def setPriorities(self):
        locprio = {}
        for prio, states in self.pomdp.prio.items():
            for state in states:
                if state not in locprio:
                    locprio[state] = prio
                else:
                    locprio[state] = max(locprio[state], prio)

        for i, o in enumerate(self.states):
            self.prio[i] = max([locprio[s] for s in o])

    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.actions = pomdp.actions
        self.actionsinv = pomdp.actionsinv
        self.trans = {}
        self.prio = {}
        # states will be belief supports, but to
        # create them it's best to do a foward
        # exploration
        self.start = 0
        st = tuple([k for k in pomdp.start
                    if pomdp.start[k] > 0])
        self.states = [st]
        self.statesinv = {st: 0}
        explore = [st]
        while len(explore) > 0:
            st = explore.pop()
            self.trans[self.statesinv[st]] = {}
            for i, _ in enumerate(self.actions):
                self.trans[self.statesinv[st]][i] = []
                succ = []
                for src in st:
                    succ.extend([k for k in self.pomdp.trans[src][i]
                                 if self.pomdp.trans[src][i][k] > 0])
                succ = set(succ)
                beliefs = {}
                for dst in succ:
                    for o, p in self.pomdp.obsfun[i][dst].items():
                        if p > 0:
                            if o in beliefs:
                                beliefs[o].append(dst)
                            else:
                                beliefs[o] = [dst]
                for o, belief in beliefs.items():
                    belief = tuple(belief)
                    if belief in self.statesinv:
                        idbf = self.statesinv[belief]
                    else:
                        idbf = len(self.states)
                        self.statesinv[belief] = len(self.states)
                        self.states.append(belief)
                        explore.append(belief)
                    self.trans[self.statesinv[st]][i].append(idbf)

    def show(self, outfname, reach=None, reachStrat=None,
             goodMecs=None, goodStrat=None, greatMecs=None, greatStrat=None, mecs_and_strats=None):

        if mecs_and_strats is not None:
            raise NotImplementedError

        # I am assuming here that reach and what follows are either all None
        # or all not None
        if goodMecs is not None:
            allgood = set().union(*goodMecs)
            allgreat = set().union(*greatMecs)
        G = pgv.AGraph(directed=True, strict=False)
        for i, s in enumerate(self.states):
            name = self.prettyName(s)
            if i in self.prio:
                G.add_node(i, label=f"{name} : {self.prio[i]}")
            else:
                G.add_node(i, label=name)
        for src, _ in enumerate(self.states):
            for a, act in enumerate(self.actions):
                color = "black"
                if reach is not None:
                    if src in allgreat and a in greatStrat[src]:
                        color = "green"
                    elif src in allgood and a in goodStrat[src]:
                        color = "yellow"
                    elif src in reach and a in reachStrat[src]:
                        color = "blue"
                for dst in self.trans[src][a]:
                    G.add_edge(src, dst, color=color, label=f"{act}")
        G.layout("dot")
        G.draw(outfname)
