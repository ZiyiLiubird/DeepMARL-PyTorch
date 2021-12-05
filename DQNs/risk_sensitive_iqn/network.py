from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DQNBase(nn.Module):

    def __init__(self, num_channels, embedding_dim=512):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.embedding_dim = embedding_dim


    def forward(self, states):
        
        batch_size = states.shape[0]
        # Calculate embedding of states.
        state_embedding = self.net(states)
        assert state_embedding.shape == (batch_size, self.embedding_dim)

        return state_embedding


class CosineEmbeddingNetwork(nn.Module):

    def __init__(self, num_cosines=64, embedding_dim=512, noisy_net=False):
        super(CosineEmbeddingNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):

        batch_size = taus.shape[0]
        N = taus.shape[1]
        i_pi = torch.arange(
           start=1, end=self.num_cosines+1, dtype=taus.dtype,
           device=taus.device).view(1, 1, self.num_cosines)

        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size, N, self.num_cosines)

        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)

        return tau_embeddings


class QuantileNetwork(nn.Module):

    def __init__(self, act_dim, embedding_dim):
        super(QuantileNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )
        self.embedding_dim = embedding_dim
        self.act_dim = act_dim


    def forward(self, state_embedding, tau_embedding):
        
        assert state_embedding.shape[0] == tau_embedding.shape[0]
        assert state_embedding.shape[1] == tau_embedding.shape[2]
        batch_size = state_embedding.shape[0]
        N = tau_embedding.shape[1]

        state_embedding = state_embedding.view(batch_size, 1, self.embedding_dim)

        embeddings = (state_embedding * tau_embedding).view(
            batch_size, N, self.embedding_dim
        )
        quantiles = self.net(embeddings)

        return quantiles.view(batch_size, N, self.act_dim)
