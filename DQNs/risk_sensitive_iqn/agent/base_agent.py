from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import gym
from torch.cuda import is_available
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from memory.replay_memory import ReplayBuffer
from utils import LinearAnneaer



class BaseAgent(ABC):

    def __init__(self, env:gym.Env, test_env:gym.Env, log_dir, num_steps=4.5*(10**4),
                 batch_size=64, memory_size=10**4, gamma=0.99, multi_step=1,
                 update_interval=1, target_update_interval=1000,
                 start_steps=1000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=35000, double_q_learning=True,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=5000,
                 max_episode_steps=500, grad_cliping=5.0, cuda=True, seed=0, tau=0.01):

        """
        env and test_env are defined in Agent.

        Args:
            update_interval: 
            target_update_interval:
            eval_interval:
            num_eval_steps:
        """

        self.env = env
        self.test_env = test_env
        self.obs_dim = env.observation_space.shape[0]
        torch.manual_seed(seed=seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2**31-1-seed)
        self.device = torch.device(
            "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
        )

        self.online_net = None
        self.target_net = None
        self.tau = tau

        self.memory = ReplayBuffer(size=memory_size)
        # self.log_dir = log_dir
        # self.model_dir = os.path.join(log_dir, 'model')
        # self.summary_dir = os.path.join(log_dir, 'summary')
        # if not os.path.exists(self.model_dir):
        #     os.makedirs(self.model_dir)
        # if not os.path.exists(self.summary_dir):
        #     os.makedirs(self.summary_dir)

        # self.writer = SummaryWriter(log_dir=self.summary_dir)
        # self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.act_dim = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning
        self.dueling_q_learning = dueling_net
        self.noisy_net = noisy_net
        self.use_per = use_per

        # self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps
        )
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping
        self.global_returns = []

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break
        return self.global_returns

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_random(self, eval=False):
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        if self.noisy_net:
            return False
        return np.random.rand() < self.epsilon_train.get()

    def update_target(self, update_net:nn.Module, pred_net: nn.Module, tau):
        for update_param, pred_param in zip(update_net.parameters(), pred_net.parameters()):
            update_param.data.copy_((1 - tau) * update_param.data + tau * pred_param.data)


    def explore(self):
        # Act with randomness
        action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        # Act withoud randomness
        state = torch.FloatTensor(state).view(-1, self.obs_dim).to(self.device)
        with torch.no_grad():
            action = self.online_net.calculate_q(state).argmax().item()
        return action

    @abstractmethod
    def learn(self):
        pass

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth')
        )
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth')
        )

    def load_models(self, save_dir):
        self.online_net.state_dict(
            torch.load(os.path.join(save_dir, 'online_net.pth'))
        )
        self.target_net.state_dict(
            torch.load(os.path.join(save_dir, 'target_net.pth'))
        )

    def train_episode(self):
        '''
        Rollout one episode
        '''
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            
            if self.is_random(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()
        print(f'episode returns: {episode_return}  || steps: {self.steps}')
        self.global_returns.append(episode_return)

        
    def train_step_interval(self):
        self.epsilon_train.step()

        # if self.steps % self.target_update_interval == 0:
        self.update_target(self.target_net, self.online_net, self.tau)
        if self.is_update():
            self.learn()




