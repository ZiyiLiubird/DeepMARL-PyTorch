import torch
from torch.optim import Adam
import gym
from torch.optim.rmsprop import RMSprop

from model.fqf import FQF
from utils import calculate_quantile_huber_loss, disable_gradients, evaluate_quantile_at_action, update_params

from agent.base_agent import BaseAgent


class FQFAgent(BaseAgent):

    def __init__(self, env: gym.Env, test_env: gym.Env, log_dir=None, num_steps=4.5 * (10 ** 4),
                 batch_size=64, N=64, num_cosines=64, ent_coef=0,
                 kappa=1.0, quantile_lr=1e-3, fraction_lr=1e-5, memory_size=10 ** 4, gamma=0.99, multi_step=1, update_interval=2,
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

        self.online_net = FQF(num_channels=env.observation_space.shape[0],
        act_dim=self.act_dim, N=N, num_cosines=num_cosines, dueling_net=dueling_net,
        noisy_net=noisy_net).to(self.device)

        self.target_net = FQF(
            num_channels=env.observation_space.shape[0], act_dim=self.act_dim,
            N=N, num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net, target=True
        ).to(self.device)

        self.update_target(self.target_net, self.target_net, 1.0)
        disable_gradients(self.target_net)

        # self.fraction_optim = RMSprop(
        #     self.online_net.fraction_net.parameters(),
        #     lr=fraction_lr, alpha=0.95, eps=0.00001
        # )

        # self.quantile_optim = Adam(
        #     list(self.online_net.dqn_net.parameters())
        #     + list(self.online_net.cosine_net.parameters())
        #     + list(self.online_net.quantile_net.parameters()),
        #     lr=quantile_lr, eps=1e-2/batch_size
        # )

        self.optim = Adam(
            self.online_net.parameters(), lr=quantile_lr
        )


        self.ent_coef = ent_coef
        self.N = N
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
        
        # Calculate fractions of current states and entropies.
        taus, tau_hats, entropies =\
            self.online_net.calculate_fractions(
                state_embeddings=state_embeddings.detach())

        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantile_hats = evaluate_quantile_at_action(
            self.online_net.calculate_quantiles(
                tau_hats, state_embeddings=state_embeddings),
            actions)
        assert current_sa_quantile_hats.shape == (
            self.batch_size, self.N, 1)


        # NOTE: Detach state_embeddings not to update convolution layers. Also,
        # detach current_sa_quantile_hats because I calculate gradients of taus
        # explicitly, not by backpropagation.
        fraction_loss = self.calculate_fraction_loss(
            state_embeddings.detach(), current_sa_quantile_hats.detach(),
            taus, actions, weights)

        quantile_loss, mean_q, errors = self.calculate_quantile_loss(
            state_embeddings, tau_hats, current_sa_quantile_hats, actions,
            rewards, next_states, dones, weights)
        assert errors.shape == (self.batch_size, 1)

        entropy_loss = -self.ent_coef * entropies.mean()

        # update_params(
        #     self.fraction_optim, fraction_loss + entropy_loss,
        #     networks=[self.online_net.fraction_net], retain_graph=True,
        #     grad_cliping=self.grad_cliping)
        # update_params(
        #     self.quantile_optim, quantile_loss,
        #     networks=[
        #         self.online_net.dqn_net, self.online_net.cosine_net,
        #         self.online_net.quantile_net],
        #     retain_graph=False, grad_cliping=self.grad_cliping)

        update_params(
            self.optim, quantile_loss + entropy_loss,
            networks=[
                self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)


    def calculate_fraction_loss(self, state_embeddings, sa_quantile_hats, taus,
                                actions, weights):
        assert not state_embeddings.requires_grad
        assert not sa_quantile_hats.requires_grad

        batch_size = state_embeddings.shape[0]

        with torch.no_grad():
            sa_quantiles = evaluate_quantile_at_action(
                self.online_net.calculate_quantiles(
                    taus=taus[:, 1:-1], state_embeddings=state_embeddings),
                actions)
            assert sa_quantiles.shape == (batch_size, self.N-1, 1)

        # NOTE: Proposition 1 in the paper requires F^{-1} is non-decreasing.
        # I relax this requirements and calculate gradients of taus even when
        # F^{-1} is not non-decreasing.

        values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
        signs_1 = sa_quantiles > torch.cat([
            sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
        assert values_1.shape == signs_1.shape

        values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
        signs_2 = sa_quantiles < torch.cat([
            sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
        assert values_2.shape == signs_2.shape

        gradient_of_taus = (
            torch.where(signs_1, values_1, -values_1)
            + torch.where(signs_2, values_2, -values_2)
        ).view(batch_size, self.N-1)
        assert not gradient_of_taus.requires_grad
        assert gradient_of_taus.shape == taus[:, 1:-1].shape

        # Gradients of the network parameters and corresponding loss
        # are calculated using chain rule.
        if weights is not None:
            fraction_loss = ((
                (gradient_of_taus * taus[:, 1:-1]).sum(dim=1, keepdim=True)
            ) * weights).mean()
        else:
            fraction_loss = \
                (gradient_of_taus * taus[:, 1:-1]).sum(dim=1).mean()

        return fraction_loss


    def calculate_quantile_loss(self, state_embeddings, tau_hats,
                                current_sa_quantile_hats, actions, rewards,
                                next_states, dones, weights):
        assert not tau_hats.requires_grad

        with torch.no_grad():
            # NOTE: Current and target quantiles share the same proposed
            # fractions to reduce computations. (i.e. next_tau_hats = tau_hats)

            # Calculate Q values of next states.
            if self.double_q_learning:
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
                self.online_net.sample_noise()
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_state_embeddings =\
                    self.target_net.calculate_state_embedding(next_states)
                next_q = self.target_net.calculate_q(
                    state_embeddings=next_state_embeddings,
                    fraction_net=self.online_net.fraction_net)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate features of next states.
            if self.double_q_learning:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)

            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantile_hats = evaluate_quantile_at_action(
                self.target_net.calculate_quantiles(
                    taus=tau_hats, state_embeddings=next_state_embeddings),
                next_actions).transpose(1, 2)
            assert next_sa_quantile_hats.shape == (
                self.batch_size, 1, self.N)

            # Calculate target quantile values.
            target_sa_quantile_hats = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantile_hats
            assert target_sa_quantile_hats.shape == (
                self.batch_size, 1, self.N)

        td_errors = target_sa_quantile_hats - current_sa_quantile_hats
        assert td_errors.shape == (self.batch_size, self.N, self.N)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, tau_hats, weights, self.kappa)

        return quantile_huber_loss, next_q.detach().mean().item(), \
            td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)



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
