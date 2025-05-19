#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pomdpy.env import Env
from pomdpy.parsers import pomdp
import seaborn as sns
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO


def learn(filename, buchi, cobuchi):
    print(f"Buchi states: {buchi}")
    print(f"coBuchi states: {cobuchi}")
    with open(filename) as f:
        env = pomdp.parse(f.read())
    env = Env(env, buchi, cobuchi)
    data = []
    numiter = 500
    horizon = 500
    totstep = 10_000
    modprnt = 100

    # Pierre's policy
    model = env.synthesis()
    pol = "Our algo"

    def runsims(pol, model):
        print(f"== Start simulations of {pol} ==")
        for snum in range(numiter):
            if snum % modprnt == 0:
                print(f"Starting simulation {snum + 1}")
            (obs, info) = env.reset()
            for i in range(horizon):
                action = model.predict(obs)[0]
                (obs, reward, term, trunc, info) = env.step(action)
                data.append(tuple([pol, i, info["untrumped_odd_steps"]]))
        print(f"== All simulations of {pol} done! ==")
        env.close()

    runsims(pol, model)

    # PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=totstep)
    runsims("PPO", model.get_env())

    # DQN
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=totstep)
    runsims("DQN", model.get_env())

    # A2C
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=totstep)
    runsims("A2C", model.get_env())

    # Now preparing to plot
    df = pd.DataFrame(data, columns=["Policy", "Step", "Untrumped Odd Steps"])
    sns.relplot(data=df,
                x="Step",
                y="Untrumped Odd Steps",
                kind="line",
                hue="Policy",
                style="Policy")
    plt.show()


# Set up the parse arguments
parser = argparse.ArgumentParser(
                    prog="learn.py",
                    description="Learns for the (PO)MDP")
parser.add_argument("filename",            # positional argument
                    help="POMDP filename")
parser.add_argument('-1', '--cobuchi',     # list of targets
                    nargs='+',
                    help='List of priority-1 states',
                    required=False,
                    default=[])
parser.add_argument('-2', '--buchi',       # list of targets
                    nargs='+',
                    help='List of priority-2 states',
                    required=False,
                    default=[])
if __name__ == "__main__":
    args = parser.parse_args()
    learn(args.filename, args.buchi, args.cobuchi)
    exit(0)
