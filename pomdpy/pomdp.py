from itertools import product
import numpy as np
import pygraphviz as pgv


class POMDP:
    def __init__(self):
        self.states = []
        self.actions = []
        self.obs = []
        self.statesinv = {}
        self.actionsinv = {}
        self.obsinv = {}
        # transition and obs functions
        self.start = {}
        self.trans = {}
        self.obsfun = {}
        # simulation state
        self.curstate = None

    def reset(self):
        assert len(self.states) > 0
        assert sum(self.start.values()) == 1.0
        self.curstate = np.random.choice(list(self.start.keys()),
                                         p=list(self.start.values()))

    def step(self, action):
        assert len(self.states) > 0
        assert self.curstate is not None
        if not isinstance(action, int):
            action = self.actionsinv[action]
        distr = self.trans[self.curstate][action]

        # we need to check whether we have a pseudo distribution
        s = sum(distr.values())
        assert s == 1.0

        # ready to sample state
        self.curstate = np.random.choice(list(distr.keys()),
                                         p=list(distr.values()))

        # ready to sample observation now
        distr = self.obsfun[action][self.curstate]
        s = sum(distr.values())
        assert s == 1.0
        nextobs = np.random.choice(list(distr.keys()),
                                   p=list(distr.values()))
        return self.obs[nextobs]

    def setUniformStart(self, inc=None, exc=None):
        assert len(self.states) > 0
        if inc is None and exc is None:
            inc = list(range(len(self.states)))
        elif exc is not None:
            assert inc is None
            if isinstance(exc, str):
                inc = [i for i, s in enumerate(self.states) if s != exc]
            else:  # it's a list then
                inc = [i for i, s in enumerate(self.states) if s not in exc]
        elif inc is not None:
            assert exc is None
            if isinstance(inc, str):
                inc = [self.statesinv[inc]]
            else:  # it's a list then
                inc = [self.statesinv[i] if isinstance(i, str)
                       else i
                       for i in inc]
        else:
            assert False  # Both can't be true
        for i in inc:
            self.start[i] = 1.0 / len(inc)

    def _addOneTrans(self, src, act, dst, p):
        if src not in self.trans:
            self.trans[src] = {}
        if act not in self.trans[src]:
            self.trans[src][act] = {}
        self.trans[src][act][dst] = p

    def _addOneUniformTrans(self, act, src):
        for i, _ in enumerate(self.states):
            self._addOneTrans(src, act, i, 1.0 / len(self.states))

    def _checkAct(self, act):
        if act is None:
            act = list(range(len(self.actions)))
        elif not isinstance(act, int):
            act = [self.actionsinv[act]]
        else:
            assert act >= 0 and act < len(self.actions)
            act = [act]
        return act

    def _checkState(self, src):
        if src is None:
            src = list(range(len(self.states)))
        elif not isinstance(src, int):
            src = [self.statesinv[src]]
        else:
            assert src >= 0 and src < len(self.states)
            src = [src]
        return src

    def addUniformTrans(self, act=None, src=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        src = self._checkState(src)
        for a, s in product(act, src):
            self._addOneUniformTrans(a, s)

    def addTrans(self, matrix, act=None, src=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        src = self._checkState(src)
        assert len(src) != 1 or len(matrix) == len(self.states)
        assert len(src) == 1 or\
            len(matrix) == len(self.states) * len(self.states)
        for s, a in product(src, act):
            for succ, psucc in enumerate(matrix[0:len(self.states)]):
                self._addOneTrans(s, a, succ, psucc)
            matrix = matrix[len(self.states):]

    def _addOneIdentityTrans(self, act):
        for i, _ in enumerate(self.states):
            self._addOneTrans(i, act, i, 1.0)

    def addIdentityTrans(self, act=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        for a in act:
            self._addOneIdentityTrans(a)

    def _addOneObs(self, act, dst, o, p):
        if act not in self.obsfun:
            self.obsfun[act] = {}
        if dst not in self.obsfun[act]:
            self.obsfun[act][dst] = {}
        self.obsfun[act][dst][o] = p

    def _addOneUniformObs(self, act, dst):
        for i, _ in enumerate(self.obs):
            self._addOneObs(act, dst, i, 1.0 / len(self.obs))

    def addUniformObs(self, act=None, dst=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        dst = self._checkState(dst)
        for a, s in product(act, dst):
            self._addOneUniformObs(a, s)

    def addObs(self, matrix, act=None, dst=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        dst = self._checkState(dst)
        assert len(dst) != 1 or len(matrix) == len(self.obs)
        assert len(dst) == 1 or\
            len(matrix) == len(self.obs) * len(self.states)
        for a, d in product(act, dst):
            for i, p in enumerate(matrix[0:len(self.obs)]):
                self._addOneObs(a, d, i, p)
            matrix = matrix[len(self.obs):]

    def setStates(self, ids):
        if isinstance(ids, int):
            ids = list(map(str, range(0, ids)))
        self.states = ids
        for i, s in enumerate(ids):
            self.statesinv[s] = i

    def setActions(self, ids):
        if isinstance(ids, int):
            ids = list(map(str, range(0, ids)))
        self.actions = ids
        for i, s in enumerate(ids):
            self.actionsinv[s] = i

    def setObs(self, ids):
        if isinstance(ids, int):
            ids = list(map(str, range(0, ids)))
        self.obs = ids
        for i, s in enumerate(ids):
            self.obsinv[s] = i

    def show(self):
        G = pgv.AGraph(directed=True)
        for i, s in enumerate(self.states):
            G.add_node(i, label=s)
        for src, _ in enumerate(self.states):
            for a, act in enumerate(self.actions):
                for dst, p in self.trans[src][a].items():
                    if p > 0:
                        G.add_edge(src, dst, label=f"{act} : {p}")
        G.layout()
        G.draw("pomdp.png")
