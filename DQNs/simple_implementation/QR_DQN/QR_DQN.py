import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import os
from os.path import dirname
import sys

path = dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))
sys.path.append(path)

from utils import ReplayBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def target_update(target_net:nn.Module, pred_net:nn.Module, tau:float):
    for target_param, pred_param in zip(target_net.parameters(), pred_net.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * pred_param.data)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, q_num):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.q_num = q_num
        self.fc1 = nn.Linear(self.obs_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.act_dim * self.q_num)


    def forward(self, x):
        
        x = x.view(-1, self.obs_dim)
        mb = x.shape[0]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x).view(mb, self.act_dim, self.q_num)
        return x



class QR_DQN:

    def __init__(self, obs_dim, act_dim, quantiles_target, device, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = args.lr
        self.q_num = args.q_num
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.device = device
        self.quantiles_target = quantiles_target
        self.memory_capacity = args.memory_capacity
        self.pred_net, self.target_net = Critic(obs_dim, act_dim, self.q_num).to(device), Critic(obs_dim, act_dim, self.q_num).to(device)
        target_update(self.target_net, self.pred_net, self.tau)
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.memory_capacity)
        self.memory_counter = 0
        self.learn_step_counter = 0

    def choose_action(self, state, epsilon):
        state = torch.FloatTensor(state).to(self.device)
        state = state.view(-1, self.obs_dim)
        mb = state.shape[0]
        if np.random.uniform() > epsilon:
        
            action_value_ = self.pred_net(state)
            action_value = torch.mean(action_value_, dim=2)
            action = torch.argmax(action_value, dim=-1).data.cpu().numpy().item()

        else:
            action = np.random.randint(0, self.act_dim)

        return action

    def store_transitions(self, s, a, r, s_, d):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(d))

    def update(self):
        self.learn_step_counter += 1
        obses, acts, rews, obs_nexts, dones = self.replay_buffer.sample(self.batch_size)
        target_update(self.target_net, self.pred_net, self.tau)
        obses = torch.FloatTensor(obses).to(self.device)
        acts = torch.LongTensor(acts).to(self.device)
        rews = torch.FloatTensor(rews).to(self.device)
        obs_nexts = torch.FloatTensor(obs_nexts).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_eval = self.pred_net(obses)
        mb_size = q_eval.shape[0]
        q_eval = torch.stack([q_eval[i].index_select(0, acts[i]) for i in range(mb_size)]).squeeze(1)
        # (m, q_num)
        q_eval = q_eval.unsqueeze(2)
        # (m, q_num, 1) dim 1 is present quantiles, 2 is target

        # get next state values
        q_next = self.target_net(obs_nexts).detach()
        best_actions = q_next.mean(dim=2).argmax(dim=1)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        # (m, q_num)
        q_target = rews.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * q_next
        q_target = q_target.unsqueeze(1)
        # (m, 1, q_num)

        # quantile huber loss
        u = q_target.detach() - q_eval
        # (m, q_num, q_num)
        tau = torch.FloatTensor(self.quantiles_target).to(self.device).view(1, -1, 1)
        weight = torch.abs(tau - u.le(0.).float())
        loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
        loss = torch.mean(weight * loss, dim=-1).mean(dim=-1)
        loss = torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def save_model(self):
        pass

    def load_model(self):
        pass




def train(args):
    env_name = args.env_name
    learn_start = args.learn_start
    q_num = args.q_num
    epoch = args.epoch
    epsilon = args.epsilon
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(env_name)
    quantiles = np.linspace(0.0, 1.0, q_num + 1)[1: ]
    quantiles_target = (np.linspace(0.0, 1.0, q_num + 1)[: -1] + quantiles) / 2.
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = QR_DQN(obs_dim=obs_dim, act_dim=action_dim, quantiles_target=quantiles_target, device=device, args=args)

    Gt = []
    cnt = 0
    for step in range(epoch):

        obs = env.reset()
        obs = np.array(obs)
        gt = 0
        while True:
            cnt += 1
            action = agent.choose_action(obs, epsilon)
            obs_next, rew,  done, _ = env.step(action)
            agent.store_transitions(obs, action, rew, obs_next, done)
            gt += rew
            obs = obs_next
            if agent.memory_counter >= learn_start:
                agent.update()
            if done:
                print(f'epoch {step} cumulative rewards: {gt}')
                break

        if step <= int(epoch * 0.75):
            epsilon -= 0.8/(epoch * 0.75)
        else:
            epsilon -= 0.19/(epoch * 0.25)
        Gt.append(gt)
    vis(Gt, env_name)


def vis(Gt, env_name="CartPole-v0"):
    plt.plot(Gt)
    plt.xlabel("episodes")
    plt.ylabel("cumulative rewards")
    plt.title(f"QR-DQN-{env_name}")
    plt.legend()
    plt.savefig(f"QR-DQN-{env_name}.png", dpi=300)


def test(agent: QR_DQN, env, render=False):

    epsilon = 0.01
    Gt = []
    for i in range(100):
        obs = env.reset()
        rews = 0
        while True:
            act = agent.choose_action(obs, epsilon)
            obs_next, rew, done, _ = env.step(act)
            rews += rew
            obs = obs_next
            if render:
                env.render()
            if done:
                break

        Gt.append(rews)

    print(f'average rewards in 100 trils: {np.mean(Gt)}')





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
    parser.add_argument('--capacity', default=int(1e5), type=int)
    parser.add_argument('--c', default=10.0, type=float)
    args = parser.parse_args()

    train(args)

