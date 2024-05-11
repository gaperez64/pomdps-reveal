# This is an automaton that captures the belief support
# behavior of a given POMDP
class BeliefSuppAut:
    def _stateName(self, st):
        if isinstance(st, str) and st == "_sink":
            return tuple(["_sink"])
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
                succ = []
                for src in st:
                    succ.extend([k for k in self.pomdp.trans[src][i]
                                 if self.pomdp.trans[src][i][k] > 0])
                succ = tuple(set(succ))
                # TODO: this has to be further split based on observations!
                self.statesinv = {succ: len(self.states)}
                self.trans[self.statesinv[st]][i] = len(self.states)
                self.states.append(succ)
                explore.append(succ)
