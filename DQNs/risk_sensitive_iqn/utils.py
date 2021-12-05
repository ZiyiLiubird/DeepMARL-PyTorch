from collections import deque
import numpy as np
import torch
from torch.autograd import grad


def update_params(optim: torch.optim.Adam, loss, networks, retain_graph=False, grad_cliping=None):

    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    if grad_cliping:
        for net in networks:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)

    optim.step()


def evaluate_quantile_at_action(quantiles, actions):
    assert quantiles.shape[0] == actions.shape[0]

    batch_size = quantiles.shape[0]
    N = quantiles.shape[1]
    action_index = actions[..., None].expand(batch_size, N, 1)
    sa_quantiles = quantiles.gather(dim=2, index=action_index)
    return sa_quantiles.view(batch_size, N, 1)

def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
    # assert not taus.require_grad

    batch_size, N, N_dash = td_errors.shape

    element_wise_huber_loss = torch.where(td_errors<=kappa, 0.5*td_errors.pow(2), kappa*(td_errors.abs() - 0.5*kappa))
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash
    )
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - td_errors.le(0).float()
    ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash
    )
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss

class LinearAnneaer:

    def __init__(self, start_value, end_value, num_steps):
        assert num_steps > 0 and isinstance(num_steps, int)

        self.steps = 0
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

        self.a = (self.end_value - self.start_value) / self.num_steps
        self.b = self.start_value

    def step(self):
        self.steps = min(self.num_steps, self.steps + 1)

    def get(self):
        assert 0 < self.steps <= self.num_steps
        return self.a * self.steps + self.b

def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False
