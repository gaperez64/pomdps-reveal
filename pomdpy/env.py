import gymnasium as gym
import numpy as np


# This is a very thin wrapper around the POMDP for simulation matters and for
# training RL policies. Intituitively, it exposes the belief-support
# automaton via a black-box interface. That is also the main difference with
# the built-in simulation methods of POMDP, which output raw observations.
class Env(gym.Env):
    def __init__(self, pomdp, buchi, cobuchi):
        self.pomdp = pomdp
        self.actions = pomdp.actions
        self.actionsinv = pomdp.actionsinv
        self.untrumpdCnt = 0
        # we will use a one-hot encoding for subsets of states, i.e.
        # an observation support
        self.observation_space = gym.spaces.MultiBinary(len(pomdp.states))
        self.action_space = gym.spaces.Discrete(len(pomdp.actions))
        # we store the (co)buchi ids too
        self.cobuchiIds = []
        for t in cobuchi:
            if t not in self.pomdp.statesinv:
                print(f"ERROR: Could not find state {t}")
                exit(1)
            self.cobuchiIds.append(self.pomdp.statesinv[t])
        self.buchiIds = []
        for t in buchi:
            if t not in self.pomdp.statesinv:
                print(f"ERROR: Could not find state {t}")
                exit(1)
            self.buchiIds.append(self.pomdp.statesinv[t])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.untrumpdCnt is not None:
            self.untrumpdCnt = 0
        initStates = list(self.pomdp.start.keys())
        self.curstate = self.np_random.choice(
            initStates,
            p=list(self.pomdp.start.values()))
        self.curbelief = np.array([int(i in initStates)
                                   for i, _ in enumerate(self.pomdp.states)],
                                  dtype=int)
        info = {"state": self.curstate}
        return self.curbelief, info

    def step(self, action):
        distr = self.pomdp.trans[self.curstate][action]
        posSucc = [k for p, b in enumerate(self.curbelief)
                   for k in self.pomdp.trans[p][action]
                   if b == 1 and self.pomdp.trans[p][action][k] > 0]
        posSucc = set(posSucc)

        # ready to sample state
        self.curstate = self.np_random.choice(list(distr.keys()),
                                              p=list(distr.values()))
        # ready to sample observation now
        distr = self.pomdp.obsfun[action][self.curstate]
        nextobs = self.np_random.choice(list(distr.keys()),
                                        p=list(distr.values()))

        # now we need to produce the belief support, that's the
        # set of all states that can get the same observation with
        # the same action
        posSucc = [q for q in posSucc
                   if self.pomdp.obsfun[action][q][nextobs] > 0]
        self.curbelief = np.array([int(i in posSucc)
                                   for i, _ in enumerate(self.pomdp.states)],
                                  dtype=int)

        # finally, we cook a reward based on the type of state reached
        if any([x in self.buchiIds for x in posSucc]):
            priority = 2
            # FIXME: How large do we want this reward? hardcoded?
            # we do want to have any one such reward compensate
            # for all negative ones between it and the previous
            # positive one...
            reward = 100
            self.untrumpdCnt = 0
        elif any([x in self.cobuchiIds for x in posSucc]):
            priority = 1
            reward = -1
            self.untrumpdCnt += 1
        else:
            priority = 0
            reward = 0

        # defaults
        terminated = False
        truncated = False

        # for information, we also send the observation signal
        info = {"observation": self.pomdp.obs[nextobs],
                "state": self.pomdp.states[self.curstate],
                "priority": priority,
                "reward": reward,
                "untrumped_odd_steps": self.untrumpdCnt}
        return (self.curbelief, reward, terminated, truncated, info)

    def render(self):
        assert False  # not implemented, so it should not be called
