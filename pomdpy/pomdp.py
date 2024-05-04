from itertools import product


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

    def setUniformStart(self):
        assert len(self.states) > 0
        for i, _ in enumerate(self.states):
            self.start[i] = 1.0 / len(self.states)

    def _addOneUniformTrans(self, act, src):
        for i, _ in enumerate(self.states):
            if src not in self.trans:
                self.trans[src] = {}
            if act not in self.trans[src]:
                self.trans[src][act] = {}
            self.trans[src][act][i] = 1.0 / len(self.states)

    def _checkAct(self, act):
        if act is None:
            act = list(range(len(self.actions)))
        elif act is not isinstance(act, int):
            act = [self.actionsinv[act]]
        else:
            assert act >= 0 and act < len(self.actions)
            act = [act]
        return act

    def _checkState(self, src):
        if src is None:
            src = list(range(len(self.states)))
        elif src is not isinstance(src, int):
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

    def _addOneIdentityTrans(self, act):
        for i, _ in enumerate(self.states):
            if i not in self.trans:
                self.trans[i] = {}
            if act not in self.trans[i]:
                self.trans[i][act] = {}
            self.trans[i][act][i] = 1.0

    def addIdentityTrans(self, act=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        for a in act:
            self._addOneIdentityTrans(a)

    def _addOneUniformObs(self, act, dst):
        for i, _ in enumerate(self.obs):
            if act not in self.obsfun:
                self.obsfun[act] = {}
            if dst not in self.obsfun[act]:
                self.obsfun[act][dst] = {}
            self.obsfun[act][dst][i] = 1.0 / len(self.obs)

    def addUniformObs(self, act=None, dst=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        dst = self._checkState(dst)
        for a, s in product(act, dst):
            self._addOneUniformObs(a, s)

    def _addObsRow(self, act, dst, row):
        for i, p in enumerate(row):
            if act not in self.obsfun:
                self.obsfun[act] = {}
            if dst not in self.obsfun[act]:
                self.obsfun[act][dst] = {}
            self.obsfun[act][dst][i] = p

    def addObs(self, matrix, act=None, dst=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        dst = self._checkState(dst)
        assert len(dst) != 1 or len(matrix) == len(self.obs)
        assert len(dst) == 1 or\
            len(matrix) == len(self.obs) * len(self.states)
        for a, d in product(act, dst):
            self._addObsRow(a, d, matrix[0:len(self.obs)])
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
