import argparse
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
print(path)
sys.path.append(path)

import torch
import gym
from agent.iqn_agent import IQNAgent
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--env_name', default="CartPole-v0", type=str)
    parser.add_argument('--q_num', default=200, type=int)
    parser.add_argument('--learn_start', default=int(1e3), type=int)
    parser.add_argument('--memory_capacity', default=int(1e5), type=int)
    parser.add_argument('--learn_freq', default=4, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--tau', default=0.01, type=float)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    model = IQNAgent(env, env)
    results = model.run()
    label = "Risk-Averse-CVar(0.5)_IQN"
    plt.plot(results, label=label)
    plt.xlabel("episodes")
    plt.ylabel("Returns")
    plt.legend()
    plt.savefig(f'{label}.png', dpi=300)
