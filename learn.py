#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pomdpy.parsers import pomdp
from pomdpy.env import Env
import seaborn as sns
from stable_baselines3 import PPO


def learn(filename, buchi, cobuchi):
    with open(filename) as f:
        env = pomdp.parse(f.read())
    env = Env(env, buchi, cobuchi)

    print("Start learning")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    print("Learning done!")

    vec_env = model.get_env()
    print("== Start simulations ==")
    data = []
    pol = "PPO"
    for snum in range(500):
        print(f"Starting simulation {snum + 1}")
        obs = vec_env.reset()
        for i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            data.append(tuple([pol, i, info[0]["untrumped_odd_steps"]]))
            assert not done
    print("== All simulations done! ==")
    env.close()

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
    print(f"file = {args.filename}, "
          "buchi = {args.buchi}, cobuchi = {args.cobuchi}")
    learn(args.filename, args.buchi, args.cobuchi)
    exit(0)
