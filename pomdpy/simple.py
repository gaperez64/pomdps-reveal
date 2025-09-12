from pomdpy import POMDP


# This is essentially Pierre's version of a POMDP where the transitions spit
# out a signal and an observation simultaneously
class SimplePOMDP(POMDP):
    def __init__(self, other: POMDP):
        self.states = other.states
        self.actions = other.actions
        self.obs = other.obs
        self.prio = other.prio
        self.statesinv = other.statesinv
        self.actionsinv = other.actionsinv
        self.obsinv = other.obsinv
        # simulation state
        self.curstate = None

        # transition function, simplified
        self.start = other.start
        # NOTE: We want a distribution over States x Observations
        # and not two split distributions, one over states and one over
        # observations
        # FIXME: We are assuming that states and observations are numbers or
        # at least hashable since a tuple of them is being used as a key for a
        # dictionary
        self.trans = {}
        for src in other.trans:
            self.trans[src] = {}
            for act in other.trans[src]:
                self.trans[src][act] = {}
                for dst in other.trans[src][act]:
                    for o in other.obsfun[act][dst]:
                        self.trans[src][act][(dst, o)] =\
                            other.trans[src][act][dst] *\
                            other.obsfun[act][dst][o]
