"""
Neural network definitions for PID-Lagrangian SAC.

Architecture (per paper):
- Actor:   state -> [256, ReLU] -> [256, ReLU] -> (mu, log_sigma)
- Critic:  state + action -> [256, ReLU] -> [256, ReLU] -> Q
- Total 6 networks: 2 reward critics + 2 safety critics + 1 actor + 1 value (targets via polyak)
"""
import torch
import torch.nn as nn
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class Actor(nn.Module):
    """Gaussian policy network.

    Outputs mean and log_std for each action dimension.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        # Log std bounds (SB3-style)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.apply(init_weights)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, deterministic=False):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            return mu, torch.tensor(0.0, device=obs.device)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        # Tanh squashing
        action = torch.tanh(u)
        # Log prob correction for tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def log_prob_from_action(self, obs, action):
        """Compute log_prob of a given action (for actor loss)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        # Invert tanh
        u = torch.atanh(torch.clamp(action, -1 + 1e-6, 1 - 1e-6))
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return log_prob


class Critic(nn.Module):
    """Q-network: state + action -> Q-value."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_weights)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)
