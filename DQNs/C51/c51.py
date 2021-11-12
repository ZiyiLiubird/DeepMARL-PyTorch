import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayBuffer

import random
import os
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

seed = 0
np.random.seed(seed=seed)
torch.manual_seed(seed)

'''DQN settings'''
TARGET_REPLACE_ITER = 10
LEARN_START = int(1e+3)
MEMORY_CAPACITY = int(1e+4)
LEARN_FREQ = 1
N_ATOM = 51

'''Environment Settings'''
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
action_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]
print(f'obs_dim: {obs_dim}')
print(f'act_dim: {action_dim}')
# prior knowledge of return distributions
V_MIN = -100.
V_MAX = 100.
V_RANGE = np.linspace(V_MIN, V_MAX, N_ATOM)
V_STEP = ((V_MAX - V_MIN) / (N_ATOM - 1))
# STEP_NUM = int(1e6)
GAMMA = 0.99
RENDERING = False

'''Training settings'''
USE_GPU = torch.cuda.is_available()
print(f'USE GPU: {USE_GPU}')
batch_size = 64
lr = 1e-3
epsilon = 0.9

'''Save&Load Settings'''
# check save/load
save = True
load = False
save_freq = int(1e3)
# paths for predction net, target net, result log
PRED_PATH = './data/model/C51_pred_net_' + ENV_NAME + '.pkl'
TARGET_PATH = './data/model/C51_target_net_' + ENV_NAME + '.pkl'
RESULT_PATH = './data/plots/C51_result_' +ENV_NAME + '.pkl'


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        
        # action value distribution
        self.fc_q = nn.Linear(256, action_dim * N_ATOM)


    def forward(self, x):
        # x.size(0) : minibatch size
        x = x.view(-1, obs_dim)
        mb_size = x.size(0)
        # x.size(0) : mini-batch size
        x = x.view(x.size(0), -1)
        # print(f'x.shape{x.shape}')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # note that output of C-51 is prob mass of value distribution
        action_value = F.softmax(self.fc_q(x).view(mb_size, action_dim, N_ATOM), dim=2)

        return action_value

    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = FC(), FC()
        self.update_target(self.target_net, self.pred_net, 1.0)
        self.pred_net.cuda()
        self.target_net.cuda()
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=lr)
        self.value_range = torch.FloatTensor(V_RANGE).cuda()
        self.batch_size = batch_size
        self.gamma = GAMMA

    def update_target(self, target, pred, update_rate):
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) * target_param + update_rate * pred_param.data)

    def load_model(self):
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def choose_action(self, x, epsilon):
        x = torch.FloatTensor(x).cuda()
        # print(f'x size: {x.size(0)}')
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, action_dim)
        else:
            action = self.pred_net(x)
            action = torch.argmax(torch.sum(action * self.value_range.view(1,1,-1), dim=2), 1).data.cpu().numpy().item()
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 0.1)
        obses, acts, rews, obs_nexts, dones = self.replay_buffer.sample(self.batch_size)
        obses = torch.FloatTensor(obses).cuda()
        acts = torch.LongTensor(acts).cuda()
        obs_nexts = torch.FloatTensor(obs_nexts).cuda()
        
        q_eval = self.pred_net(obses)
        q_target = self.target_net(obs_nexts)
        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i].index_select(0, acts[i]) for i in range(mb_size)]).squeeze(1)
        # (m, N_ATOM)

        # define target distribution
        q_target = np.zeros((mb_size, N_ATOM))
        # get next state value
        q_next = self.target_net(obs_nexts).detach()
        # selsct best action values
        q_next_mean = torch.sum(q_next * self.value_range.view(1, 1, -1), 2)
        best_actions = q_next_mean.argmax(dim=1)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        q_next = q_next.data.cpu().numpy()
        # Categorical Projection
        '''
        next_v_range: values of possible return, shape: (m, N_ATOM)
        next_v_pos: 
        '''
        next_v_range = np.expand_dims(rews, 1) + self.gamma * np.expand_dims((1. - dones), 1) \
            * np.expand_dims(self.value_range.data.cpu().numpy(), 0)
        next_v_pos = np.zeros_like(next_v_range)

        next_v_range = np.clip(next_v_range, V_MIN, V_MAX)
        next_v_pos = (next_v_range - V_MIN) / V_STEP
        lb = np.floor(next_v_pos).astype(int)
        ub = np.ceil(next_v_pos).astype(int)
        for i in range(mb_size):
            for j in range(N_ATOM):
                q_target[i, lb[i, j]] += (q_next * (ub - next_v_pos))[i, j]
                q_target[i, ub[i, j]] += (q_next * (next_v_pos - lb))[i, j]

        q_target = torch.FloatTensor(q_target).cuda()

        loss = q_target * ( torch.log(q_eval + 1e-8))
        loss = - torch.mean(torch.sum(loss, dim=-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def test(env, agent:DQN, render=False):
    r = []
    for i in range(100):
        rewards = 0
        obs = env.reset()
        while True:
            act = agent.choose_action(obs, 0)
            obs_next, rew, done, _ = env.step(act)
            rewards += rew
            obs = obs_next
            if render:
                env.render()
            if done:
                break
        r.append(rewards)
    print(f'evaluation rewards: {np.mean(r)}')

dqn = DQN()
result = []
print('Initialize results!')
print('Collecting experience....')
start_time = time.time()
STEP_NUM = 200
c = 0
results = []
obs = np.array(env.reset())
for epoch in range(1, STEP_NUM + 1):
    r = 0
    while True:
        c += 1
        action = dqn.choose_action(obs, epsilon)
        obs_next, reward, done, _ = env.step(action)
        r += reward
        obs_next = np.array(obs_next)
        # for i in range(N_ENVS):
        dqn.store_transition(obs, action, reward, obs_next, done)
        
        obs = obs_next
        if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
            dqn.learn()

        if done:
            obs = np.array(env.reset())
            break
    time_interval = round(time.time() - start_time, 2)
    print(f'Used time: {time_interval} | step: {c} | epoch: {epoch}')
    print(f'cumulative rewards: {r}')
    results.append(r)
    if epoch <= int(150):
        epsilon -= 0.8/150
    elif epoch <= int(200):
        epsilon -= 0.09/50

test(env, dqn)
plt.plot(results, label="C51")
plt.xlabel('episodes')
plt.ylabel('Rewards')
plt.legend()
plt.savefig('C51.png', dpi=300)