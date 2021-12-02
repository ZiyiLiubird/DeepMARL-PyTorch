import torch
import numpy as np
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import os
from os.path import dirname
import sys

path = dirname(dirname(dirname(os.path.abspath(__file__))))
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
    def __init__(self, obs_dim, act_dim, q_num, N):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.N = N
        self.q_num = q_num
        self.fc1 = nn.Linear(self.obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.act_dim)
        self.tau_embedding = nn.Linear(self.N, 256, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(self.q_num))


    def forward(self, x, tau=None):

        x = x.view(-1, self.obs_dim)
        mb = x.shape[0]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        if tau == None:
            tau = torch.rand(self.q_num, 1).cuda()
        quants = torch.arange(0, self.N, 1.0).cuda()
        cos_trans = torch.cos(tau * quants * np.pi) # (quants, N)
        phi = F.relu(self.tau_embedding(cos_trans) + self.phi_bias.unsqueeze(1)).unsqueeze(0) # (1, quants, 256)
        x = x.view(mb, -1).unsqueeze(1) # (mb, 1, 256)
        x = F.relu(x * phi)
        x = self.fc3(x) # (mb, quants, act_dim)
        return x, tau



class IQN:

    def __init__(self, obs_dim, act_dim, device, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = args.lr
        self.q_num = args.q_num
        self.N = args.N
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.device = device
        self.memory_capacity = args.memory_capacity
        self.pred_net, self.target_net = Critic(obs_dim, act_dim, self.q_num, self.N).to(device), Critic(obs_dim, act_dim, self.q_num, self.N).to(device)
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
        
            action_value_, tau = self.pred_net(state)
            action_value = torch.mean(action_value_, dim=1)
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

        q_eval, q_eval_tau = self.pred_net(obses)
        mb_size = q_eval.shape[0]
        q_eval = torch.stack([q_eval[i].index_select(1, acts[i]) for i in range(mb_size)]).squeeze(2)
        # (m, q_num)
        q_eval = q_eval.unsqueeze(2)
        # (m, q_num, 1) dim 1 is present quantiles, 2 is target

        # get next state values
        q_next, q_next_tau = self.target_net(obs_nexts, q_eval_tau)
        q_next = q_next.detach()
        best_actions = q_next.mean(dim=1).argmax(dim=1)
        q_next = torch.stack([q_next[i].index_select(1, best_actions[i]) for i in range(mb_size)]).squeeze(2)
        # (m, q_num)
        q_target = rews.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * q_next
        q_target = q_target.unsqueeze(1)
        # (m, 1, q_num)

        # quantile huber loss
        u = q_target.detach() - q_eval
        # (m, q_num, q_num)
        tau = q_eval_tau.unsqueeze(0)
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
    epoch = args.epoch
    epsilon = args.epsilon
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(env_name)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = IQN(obs_dim=obs_dim, act_dim=action_dim, device=device, args=args)

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
        # print(f'average rewards: {np.mean(Gt)}')
    vis(Gt, env_name)


def vis(Gt, env_name="CartPole-v0"):
    plt.plot(Gt)
    plt.xlabel("episodes")
    plt.ylabel("cumulative rewards")
    plt.title(f"IQN-{env_name}")
    plt.legend()
    plt.savefig(f"IQN-{env_name}-same_tau2.png", dpi=300)


def test(agent: IQN, env, render=False):

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
    parser.add_argument('--q_num', default=128, type=int)
    parser.add_argument('--learn_start', default=int(1e3), type=int)
    parser.add_argument('--memory_capacity', default=int(1e5), type=int)
    parser.add_argument('--learn_freq', default=4, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--capacity', default=int(1e5), type=int)
    parser.add_argument('--N', default=64, type=int)


    args = parser.parse_args()
    
    train(args)

