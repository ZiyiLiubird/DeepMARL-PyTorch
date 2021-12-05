import torch
import torch.nn as nn
from network import DQNBase, CosineEmbeddingNetwork, QuantileNetwork


class IQN(nn.Module):

    def __init__(self, num_channels, act_dim, K=32, num_cosines=64,
                 embedding_dim=512, dueling_net=False, noisy_net=False):
        super(IQN, self).__init__()

        # Feature extractor of DQN
        self.dqn_net = DQNBase(num_channels=num_channels)
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=embedding_dim, noisy_net=noisy_net)
        self.quantile_net = QuantileNetwork(act_dim, embedding_dim)
        self.num_channels = num_channels
        self.act_dim = act_dim
        self.K = K
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net


    def calculate_state_embedding(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):

        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):

        assert states is not None or state_embeddings is not None

        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        taus = torch.rand(batch_size, self.K, dtype=state_embeddings.dtype,
         device=state_embeddings.device)
        
        taus = taus * 0.25

        quantiles = self.calculate_quantiles(taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.act_dim)

        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.act_dim)
        return q
        
