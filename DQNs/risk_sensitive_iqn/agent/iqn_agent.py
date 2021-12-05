import torch
from torch.optim import Adam
import gym

from model.iqn import IQN
from utils import calculate_quantile_huber_loss, disable_gradients, evaluate_quantile_at_action, update_params

from agent.base_agent import BaseAgent


class IQNAgent(BaseAgent):

    def __init__(self, env: gym.Env, test_env: gym.Env, log_dir=None, num_steps=4.5 * (10 ** 4),
                 batch_size=64, N=64, N_dash=64, K=32, num_cosines=64,
                 kappa=1.0, lr=1e-3, memory_size=10 ** 4, gamma=0.99, multi_step=1, update_interval=2,
                 target_update_interval=1000, start_steps=1000, epsilon_train=0.001, epsilon_eval=0.001,
                 epsilon_decay_steps=35000, double_q_learning=False, dueling_net=False,
                 noisy_net=False, use_per=False, log_interval=100, eval_interval=5000,
                 num_eval_steps=5000, max_episode_steps=500, grad_cliping=None, cuda=True, seed=0):

        super().__init__(env, test_env, log_dir, num_steps=num_steps,
                         batch_size=batch_size, memory_size=memory_size,
                         gamma=gamma, multi_step=multi_step,
                         update_interval=update_interval,
                         target_update_interval=target_update_interval,
                         start_steps=start_steps, epsilon_train=epsilon_train,
                         epsilon_eval=epsilon_eval,
                         epsilon_decay_steps=epsilon_decay_steps,
                         double_q_learning=double_q_learning, dueling_net=dueling_net,
                         noisy_net=noisy_net, use_per=use_per, log_interval=log_interval,
                         eval_interval=eval_interval, num_eval_steps=num_eval_steps,
                         max_episode_steps=max_episode_steps, grad_cliping=grad_cliping,
                         cuda=cuda, seed=seed)


        self.act_dim = env.action_space.n
        self.online_net = IQN(num_channels=env.observation_space.shape[0],
        act_dim=self.act_dim, K=K, num_cosines=num_cosines, dueling_net=dueling_net,
        noisy_net=noisy_net).to(self.device)

        self.target_net = IQN(
            num_channels=env.observation_space.shape[0], act_dim=self.act_dim,
            K=K, num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net
        ).to(self.device)

        self.update_target(self.target_net, self.target_net, 1.0)
        disable_gradients(self.target_net)

        self.optimizer = Adam(self.online_net.parameters(),
                              lr=lr, eps=1e-2/batch_size)

        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.kappa = kappa

    def learn(self):

        self.learning_steps += 1
        
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        weights = None

        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device)
        actions = torch.LongTensor(actions).view(self.batch_size, -1).to(self.device)
        rewards = torch.FloatTensor(rewards).view(self.batch_size, -1).to(self.device)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, -1).to(self.device)
        dones = torch.FloatTensor(dones).view(self.batch_size, -1).to(self.device)

        # Caculating features of states
        state_embeddings = self.online_net.calculate_state_embedding(states)
        quantile_loss, mean_q, errors = self.calculate_loss(
            state_embeddings, actions, rewards, next_states, dones, weights
        )

        update_params(
            self.optimizer, quantile_loss, networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping
        )



    def calculate_loss(self, state_embeddings, actions, rewards,
                       next_states, dones, weights=None):

        batch_size = state_embeddings.shape[0]

        taus = torch.rand(batch_size, self.N, dtype=state_embeddings.dtype, device=state_embeddings.device)
        quantiles = self.online_net.calculate_quantiles(taus, state_embeddings=state_embeddings)

        current_sa_quantiles = evaluate_quantile_at_action(quantiles, actions)

        assert current_sa_quantiles.shape == (batch_size, self.N, 1)

        with torch.no_grad():
            if self.double_q_learning:
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_state_embeddings = self.target_net.calculate_state_embedding(next_states)
                next_q = self.target_net.calculate_q(state_embeddings=next_state_embeddings)

            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            if self.double_q_learning:
                next_state_embeddings = self.target_net.calculate_state_embedding(next_states)

            tau_dashes = torch.rand(
                self.batch_size, self.N_dash, dtype=state_embeddings.dtype,
                device=state_embeddings.device
            )
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net.calculate_quantiles(tau_dashes, state_embeddings=next_state_embeddings),
                next_actions
            ).transpose(1, 2)
            assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]
            ) * self.gamma_n * next_sa_quantiles

            assert target_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, taus, weights, self.kappa
        )

        return quantile_huber_loss, next_q.detach().mean().item(),\
            td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)




