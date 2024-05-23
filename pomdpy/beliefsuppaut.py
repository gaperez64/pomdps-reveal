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
                        self.pre[dst].append(preimage)
                    else:
                        self.pre[dst] = [preimage]

    def tarjanSCCs(self, states, actions):
        assert len(states) > 0
        # this is Wikipedia's version of Tarjan's algorithm but
        # made nonrecursive for Python's sake
        # NOTE: the postorder nature of the DFS made it hard to
        # make nonrecursive so lookout for bugs XD
        idx = 0
        S = []
        m = max(states)
        stateIdx = [None for _ in range(m + 1)]
        stateLow = [None for _ in range(m + 1)]
        onStack = [False for _ in range(m + 1)]
        nrChild = [0 for _ in range(m + 1)]
        toVisit = [tuple([s, None]) for s in states]
        sccDecomp = []

        # now we use the stack to sim recursive calls
        while len(toVisit) > 0:
            (q, parent) = toVisit.pop()
            alreadyDone = False
            if stateIdx[q] is None:
                stateIdx[q] = idx
                stateLow[q] = idx
                idx += 1
                S.append(q)
                onStack[q] = True
            else:
                alreadyDone = True

            # because of how we add children to the recursion-simulation
            # stack, it may happen that states have already been treated
            # so we early exit after the postprocessing we pushed inward
            if not alreadyDone:
                # consider successors of q
                succ = set()
                for a in actions[q]:
                    succ.update([dst for dst in self.trans[q][a]
                                 if dst in states])
                nrChild[q] = len(succ)
                for dst in succ:
                    toVisit.append(tuple([dst, q]))

            # postprocessing
            if parent is not None:
                if onStack[q]:
                    stateLow[parent] = min(stateLow[parent], stateLow[q])
                nrChild[parent] -= 1
                # postprocessing roots: note that we have to focus on the
                # stack simulating the path (i.e. S) and not the one being
                # used to simulate the recursive DFS (i.e. toVisit)
                if nrChild[parent] == 0 and\
                        stateLow[parent] == stateIdx[parent]:
                    scc = []
                    while True:
                        w = S.pop()
                        onStack[w] = False
                        scc.append(w)
                        if parent == w:
                            break
                    sccDecomp.append(scc)
        return sccDecomp

    def goodMECs(self, great=False):
        # This is just Algorithm 47 from Baier + Katoen's Principles of Model
        # Checking with minor modifications to account for the priority
        # function making MECs good (only priority 0 present) or great (has at
        # least one state with priority 2)
        self.resetPreAct()
        pre = self.pre
        act = {}
        for i, _ in enumerate(self.states):
            act[i] = set([a for a, _ in enumerate(self.actions)])
        if not great:
            newMECs = [set([i for i, _ in enumerate(self.states)
                            if self.prio[i] == 0])]
        else:
            newMECs = [set([i for i, _ in enumerate(self.states)])]
        if len(newMECs[0]) == 0:
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
                    if great and max([self.prio[o] for o in res]) < 2:
                        continue
                    newMECs.append(res)
        return (MECs, dict(act))

    def cannotReach(self, targets):
        assert self.pre is not None
        visited = set()
        tovisit = set(targets)
        while len(tovisit) > 0:
            q = tovisit.pop()
            if q in self.pre:
                for i, _ in self.pre[q]:
                    if i not in visited:
                        tovisit.add(i)
            visited.add(q)
        return [i for i, _ in enumerate(self.states) if i not in visited]

    def almostSureReach(self, targets):
        # this code is based on Algo. 45 from the
        # Principles of Model Checking by Baier + Katoen
        self.resetPreAct()
        removed = []
        U = set(self.cannotReach(targets))
        while True:
            R = set(U)
            while len(R) > 0:
                u = R.pop()
                for t, _ in self.pre[u]:
                    if t not in U:
                        self.act[t] -= 1
                        if self.act[t] == 0:
                            R.add(t)
                            U.add(t)
                del self.pre[u]
                removed.append(u)
            U = set(self.cannotReach(targets)) - U
            if len(U) == 0:
                break
        return ([i for i, _ in enumerate(self.states) if i not in removed],
                dict(self.pre))

    def almostSureWin(self, vis=None):
        (goodMecs, goodStrat) = self.goodMECs()
        (greatMecs, greatStrat) = self.goodMECs(great=True)
        u = set().union(*goodMecs)
        u = u.union(*greatMecs)
        (r, rPre) = self.almostSureReach(u)
        if vis is not None:
            self.show(vis, r, rPre,
                      goodMecs, goodStrat,
                      greatMecs, greatStrat)
        return r

    def setBuchi(self, buchi, cobuchi):
        cobids = []
        for t in cobuchi:
            if t not in self.pomdp.statesinv:
                print(f"ERROR: Could not find state {t}")
                exit(1)
            cobids.append(self.pomdp.statesinv[t])
        bids = []
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

    def show(self, outfname, reach=None, reachPre=None,
             goodMecs=None, goodStrat=None, greatMecs=None, greatStrat=None):
        # I am assuming here that reach and what follows are either all None
        # or all not None
        if goodMecs is not None:
            allgood = set().union(*goodMecs)
            allgreat = set().union(*greatMecs)
            allreach = set([x for pre in reachPre.values() for x in pre])
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
                    elif src in reach and (src, a) in allreach:
                        color = "blue"
                for dst in self.trans[src][a]:
                    G.add_edge(src, dst, color=color, label=f"{act}")
        G.layout("dot")
        G.draw(outfname)
