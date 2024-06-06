#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pomdpy.env import Env
from pomdpy.parsers import pomdp
import seaborn as sns
from stable_baselines3 import PPO


def learn(filename, buchi, cobuchi):
    with open(filename) as f:
        env = pomdp.parse(f.read())
    env = Env(env, buchi, cobuchi)
    data = []
    numiter = 200
    horizon = 200

    # Pierre's policy
    model = env.synthesis()
    pol = "Pierre's"
    print(f"== Start simulations of {pol} ==")
    for snum in range(numiter):
        if snum % 10 == 0:
            print(f"Starting simulation {snum + 1}")
        (obs, info) = env.reset()
        for i in range(horizon):
            action = model.predict(obs)
            (obs, reward, term, trunc, info) = env.step(action)
            data.append(tuple([pol, i, info["untrumped_odd_steps"]]))
            assert not (term or trunc)
    print(f"== All simulations of {pol} done! ==")

    # PPO
    print("Start learning PPO policy")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    print("Learning done!")

    vec_env = model.get_env()
    pol = "PPO"
    print(f"== Start simulations of {pol} ==")
    for snum in range(numiter):
        if snum % 10 == 0:
            print(f"Starting simulation {snum + 1}")
        obs = vec_env.reset()
        for i in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            (obs, reward, done, info) = vec_env.step(action)
            data.append(tuple([pol, i, info[0]["untrumped_odd_steps"]]))
            assert not done
    print(f"== All simulations of {pol} done! ==")

    # close the environment once and for all
    vec_env.close()
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
