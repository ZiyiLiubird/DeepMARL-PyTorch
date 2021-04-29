import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
ten actions [0,1,...,9],
and the reward of each action obey Gaussian distribution with a mean and 1 variance.
'''
np.random.seed(42)
Q = np.zeros(10)
N = np.ones(10, dtype=np.float32)
epoch = 1000
episode = 100
epsilon = 0.1

def bandit(a):
    reward = np.random.normal(a,1)
    return reward

def ucb_policy(t, c):
    return np.argmax(Q + c * np.sqrt(np.log(t) / N))

def eps_greedy_policy():
    if np.random.uniform(0,1) <= epsilon:
        return np.random.choice(10, 1)
    else:
        return np.argmax(Q)
rewards = []
rewards1 = np.zeros((episode, epoch))
for j in range(episode):
    Q = np.zeros(10)
    N = np.ones(10, dtype=np.float32)
    for i in range(epoch):
        action = eps_greedy_policy()
        reward = bandit(action)
        Q[action] = Q[action] + 1 / N[action] *(reward - Q[action])
        N[action] += 1
        rewards1[j][i] = reward
    print("mean reward:{}".format(np.mean(rewards1[j])))
rewards.append(rewards1)
for c in range(1, 6):
    rewards2 = np.zeros((episode, epoch))
    for j in range(episode):
        step = 0
        Q = np.zeros(10)
        N = np.ones(10, dtype=np.float32)
        for i in range(epoch):
            step += 1
            action = ucb_policy(step, c)
            reward = bandit(action)
            Q[action] = Q[action] + 1 / N[action] *(reward - Q[action])
            N[action] += 1
            rewards2[j][i] = reward
        # print("mean reward:{}".format(np.mean(rewards2[j])))
    rewards.append(rewards2)
plt.figure()
for i in range(len(rewards)):
    mean = np.mean(rewards[i], axis=0)
    if i == 0:
        plt.plot(mean, label="epsilon greedy bandit")
    else:
        plt.plot(mean, label="ucb bandit with weight {}".format(i))
plt.legend()
plt.title('10-armed bandit')
plt.xlabel("steps")
plt.ylabel("rewards")
plt.savefig("bandit.png", dpi=300)
plt.show()