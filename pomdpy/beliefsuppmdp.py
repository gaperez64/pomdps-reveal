class BeliefSuppMDP:
    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.actions = pomdp.actions
        self.actionsinv = pomdp.actionsinv
        # states will be belief supports, but to
        # create them it's best to do a foward
        # exploration
        self.states = []
        self.statesinv = {}
        self.start = {}
        self.trans = {}
