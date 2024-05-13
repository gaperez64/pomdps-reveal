import pygraphviz as pgv


# This is an automaton that captures the belief support
# behavior of a given POMDP
class BeliefSuppAut:
    def _stateName(self, st):
        return tuple([self.pomdp.states[q] for q in st])

    def resetPreAct(self):
        self.pre = {}
        self.act = {}
        for i, _ in self.states:
            self.act[i] = len(self.actions)
            for a, _ in self.actions:
                for dst in self.trans[i][a]:
                    if dst in self.pre:
                        preimage = (i, a)
                        self.pre[dst].append(preimage)
                    else:
                        self.pre[dst] = [preimage]

    def cannotReach(self, targets):
        assert self.pre is not None
        visited = set()
        tovisit = set(targets)
        while len(tovisit) > 0:
            q = tovisit.pop()
            for i, _ in self.pre[q]:
                if i not in visited:
                    tovisit.add(i)
            visited.append(q)
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
                        del self.pre[u]
                        self.act[t] -= 1
                        if self.act[t] == 0:
                            R.add(t)
                            U.add(t)
                removed.append(u)
            U = set(self.cannotReach(targets)) - U
            if len(U) == 0:
                break
        return [i for i, _ in enumerate(self.states) if i not in removed]

    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.actions = pomdp.actions
        self.actionsinv = pomdp.actionsinv
        self.trans = {}
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

    def show(self, outfname):
        G = pgv.AGraph(directed=True, strict=False)
        for i, s in enumerate(self.states):
            s = self._stateName(s)
            G.add_node(i, label=s)
        for src, _ in enumerate(self.states):
            for a, act in enumerate(self.actions):
                for dst in self.trans[src][a]:
                    G.add_edge(src, dst, label=f"{act}")
        G.layout("dot")
        G.draw(outfname)
