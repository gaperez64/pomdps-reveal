import pygraphviz as pgv


# This is an automaton that captures the belief support
# behavior of a given POMDP
class BeliefSuppAut:
    def _stateName(self, st):
        return tuple([self.pomdp.states[q] for q in st])

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
                    for o, p in self.pomdp.obsfun[i][dst]:
                        if p > 0:
                            if o in beliefs:
                                beliefs[o].append(dst)
                            else:
                                beliefs[o] = [dst]
                for o, belief in beliefs:
                    belief = tuple(belief)
                    self.statesinv = {belief: len(self.states)}
                    self.trans[self.statesinv[st]][i].append(len(self.states))
                    self.states.append(belief)
                    explore.append(belief)

    def show(self, outfname):
        G = pgv.AGraph(directed=True, strict=False)
        for i, s in enumerate(self.states):
            s = self._stateName(s)
            G.add_node(i, label=s)
        for src, _ in enumerate(self.states):
            for a, act in enumerate(self.actions):
                for dst, p in self.trans[src][a].items():
                    if p > 0:
                        G.add_edge(src, dst, label=f"{act} : {p}")
        G.layout("dot")
        G.draw(outfname)
