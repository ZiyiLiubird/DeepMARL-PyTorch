import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
ten actions,
'''
np.random.seed(42)
Q = np.zeros(10)
N = np.ones(10, dtype=np.float32)
epoch = 1000
episode = 100
epsilon = 0.1
lr = 0.1

def bandit(a):
    reward = np.random.normal(a,1)
    return reward

def policy():
    if np.random.uniform(0,1) <= epsilon:
        return np.random.choice(10, 1)
    else:
        return np.argmax(Q)

rewards = np.zeros((episode, epoch))
for j in range(episode):
    Q = np.zeros(10)
    N = np.ones(10, dtype=np.float32)
    for i in range(epoch):
        action = policy()
        reward = bandit(action)
        Q[action] = Q[action] + 1 / N[action] *(reward - Q[action])
        rewards[j][i] = reward
    print("mean reward:{}".format(np.mean(rewards[j])))

mean = np.mean(rewards, axis=0)
plt.figure()
plt.plot(mean)
plt.title('10-armed bandit')
plt.xlabel("steps")
plt.ylabel("rewards")
plt.savefig("bandit.png", dpi=300)
plt.show()