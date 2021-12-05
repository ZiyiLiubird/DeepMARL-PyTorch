import torch
import torch.nn as nn
from network import DQNBase, CosineEmbeddingNetwork, FractionProposalNetwork, QuantileNetwork


class FQF(nn.Module):

    def __init__(self, num_channels, act_dim, N=32, num_cosines=32,
                 embedding_dim=512, dueling_net=False, noisy_net=False,
                 target=False):
        super(FQF, self).__init__()

        # Feature extractor of DQN
        self.dqn_net = DQNBase(num_channels=num_channels)
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=embedding_dim, noisy_net=noisy_net)
        self.quantile_net = QuantileNetwork(act_dim, embedding_dim)
        
        if not target:
            self.fraction_net = FractionProposalNetwork(
                N=N, embedding_dim=embedding_dim
            )

        self.num_channels = num_channels
        self.act_dim = act_dim
        self.N = N
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.target = target


    def calculate_state_embedding(self, states):
        return self.dqn_net(states)

    def calculate_fractions(self, states=None, state_embeddings=None,
                            fraction_net=None):
        assert states is not None or state_embeddings is not None
        assert not self.target or fraction_net is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        fraction_net = fraction_net if self.target else self.fraction_net
        taus, tau_hats, entropies = fraction_net(state_embeddings)

        return taus, tau_hats, entropies


    def calculate_quantiles(self, taus, states=None, state_embeddings=None):

        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, taus=None, tau_hats=None, states=None,
                    state_embeddings=None, fraction_net=None):

        assert states is not None or state_embeddings is not None
        assert not self.target or fraction_net is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        batch_size = state_embeddings.shape[0]

        # Calculate fractions.
        if taus is None or tau_hats is None:
            taus, tau_hats, _ = self.calculate_fractions(
                state_embeddings=state_embeddings, fraction_net=fraction_net)

        # Calculate quantiles.
        quantile_hats = self.calculate_quantiles(
            tau_hats, state_embeddings=state_embeddings)
        assert quantile_hats.shape == (batch_size, self.N, self.act_dim)

        # Calculate expectations of value distribution.
        q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantile_hats)\
            .sum(dim=1)
        assert q.shape == (batch_size, self.act_dim)

        return q
