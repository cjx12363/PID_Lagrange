"""
PID-Lagrangian Safe SAC agent.

Implements SAC with dual critics (reward + safety) and
PID-controlled Lagrange multiplier for constraint satisfaction.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from agent.networks import Actor, Critic
from agent.pid_lagrangian import PIDLagrangianUpdater


class SafeSAC:
    """SAC with safety constraint enforced via PID-Lagrangian.

    Key differences from standard SAC:
        - Safety Critic (dual Q_C) predicts constraint cost.
        - PID controller updates lambda per episode.
        - Actor loss: alpha*log_pi - (min Q_R - lambda * min Q_C)
        - Target rescaling: 1 / (1 + lambda) applied to actor gradient.
    """

    def __init__(self, obs_dim, act_dim, config):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(config.get("device", "cpu"))
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.lr = config.get("lr", 1e-4)
        self.batch_size = config.get("batch_size", 256)
        self.gradient_steps = config.get("gradient_steps", 1)
        self.alpha_lr = config.get("alpha_lr", 3e-4)
        self.cost_scale = config.get("cost_scale", 0.1)

        # Networks
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        # Reward critics (twin)
        self.critic_R1 = Critic(obs_dim, act_dim).to(self.device)
        self.critic_R2 = Critic(obs_dim, act_dim).to(self.device)
        self.critic_R1_target = copy.deepcopy(self.critic_R1)
        self.critic_R2_target = copy.deepcopy(self.critic_R2)
        # Safety critics (twin)
        self.critic_C1 = Critic(obs_dim, act_dim).to(self.device)
        self.critic_C2 = Critic(obs_dim, act_dim).to(self.device)
        self.critic_C1_target = copy.deepcopy(self.critic_C1)
        self.critic_C2_target = copy.deepcopy(self.critic_C2)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_R_optim = Adam(
            list(self.critic_R1.parameters()) + list(self.critic_R2.parameters()),
            lr=self.lr,
        )
        self.critic_C_optim = Adam(
            list(self.critic_C1.parameters()) + list(self.critic_C2.parameters()),
            lr=self.lr,
        )

        # Entropy tuning
        self.target_entropy = -act_dim
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)

        # PID Lagrangian
        pid_config = config.get("pid", {})
        self.pid = PIDLagrangianUpdater(
            K_P=pid_config.get("K_P", 0.10),
            K_I=pid_config.get("K_I", 0.01),
            K_D=pid_config.get("K_D", 0.05),
            d=pid_config.get("d", 25.0),
        )
        self.lam = 0.0  # current Lagrange multiplier

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        """Select action given numpy observation."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(obs_t, deterministic=deterministic)
            return action.squeeze(0).cpu().numpy()

    def update_pid(self, J_C):
        """Update Lagrange multiplier after each episode. J_C = undiscounted sum of costs."""
        self.lam = self.pid.update(J_C)

    def update(self, replay_buffer):
        """Perform one gradient update from replay buffer."""
        if len(replay_buffer) < self.batch_size:
            return {}

        batch = replay_buffer.sample(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(-1).to(self.device)
        costs = torch.FloatTensor(batch["costs"]).unsqueeze(-1).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(-1).to(self.device)

        # --- Update Reward Critics ---
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            q_R1_next = self.critic_R1_target(next_obs, next_actions)
            q_R2_next = self.critic_R2_target(next_obs, next_actions)
            q_R_next = torch.min(q_R1_next, q_R2_next)
            q_R_target = rewards + self.gamma * (1 - dones) * (q_R_next - self.alpha * next_log_prob)

        q_R1 = self.critic_R1(obs, actions)
        q_R2 = self.critic_R2(obs, actions)
        loss_R = F.mse_loss(q_R1, q_R_target) + F.mse_loss(q_R2, q_R_target)

        self.critic_R_optim.zero_grad()
        loss_R.backward()
        self.critic_R_optim.step()

        # --- Update Safety Critics (x2 for faster convergence) ---
        for _ in range(2):
            with torch.no_grad():
                next_actions_s, _ = self.actor.sample(next_obs)
                q_C1_next = self.critic_C1_target(next_obs, next_actions_s)
                q_C2_next = self.critic_C2_target(next_obs, next_actions_s)
                q_C_next = torch.min(q_C1_next, q_C2_next)
                q_C_target = costs + self.gamma * (1 - dones) * q_C_next

            q_C1 = self.critic_C1(obs, actions)
            q_C2 = self.critic_C2(obs, actions)
            loss_C = F.mse_loss(q_C1, q_C_target) + F.mse_loss(q_C2, q_C_target)

            self.critic_C_optim.zero_grad()
            loss_C.backward()
            self.critic_C_optim.step()

        # --- Update Actor (with PID rescaling) ---
        actions_pi, log_prob = self.actor.sample(obs)
        q_R_pi = torch.min(self.critic_R1(obs, actions_pi), self.critic_R2(obs, actions_pi))
        q_C_pi = torch.min(self.critic_C1(obs, actions_pi), self.critic_C2(obs, actions_pi))

        # Actor loss: standard SAC (cost penalty is in augmented reward)
        # lambda * cost penalty is applied directly to reward, not via Safety Critic
        actor_loss_inner = self.alpha * log_prob - q_R_pi
        loss_actor = actor_loss_inner.mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        # --- Update Entropy (alpha) ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # --- Soft update target networks ---
        self._soft_update()

        return {
            "loss_R": loss_R.item(),
            "loss_C": loss_C.item(),
            "loss_actor": loss_actor.item(),
            "alpha": self.alpha.item(),
            "lambda": self.lam,
        }

    def _soft_update(self):
        for target, source in [
            (self.critic_R1_target, self.critic_R1),
            (self.critic_R2_target, self.critic_R2),
            (self.critic_C1_target, self.critic_C1),
            (self.critic_C2_target, self.critic_C2),
        ]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_R1": self.critic_R1.state_dict(),
                "critic_R2": self.critic_R2.state_dict(),
                "critic_C1": self.critic_C1.state_dict(),
                "critic_C2": self.critic_C2.state_dict(),
                "log_alpha": self.log_alpha.item(),
                "lam": self.lam,
                "pid_states": {
                    "I": self.pid.I,
                    "J_C_prev": self.pid.J_C_prev,
                    "iteration": self.pid.iteration,
                },
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_R1.load_state_dict(ckpt["critic_R1"])
        self.critic_R2.load_state_dict(ckpt["critic_R2"])
        self.critic_C1.load_state_dict(ckpt["critic_C1"])
        self.critic_C2.load_state_dict(ckpt["critic_C2"])
        self.log_alpha = torch.tensor(ckpt.get("log_alpha", 0.0), requires_grad=True, device=self.device)
        self.lam = ckpt.get("lam", 0.0)
        pid_states = ckpt.get("pid_states", {})
        if pid_states:
            self.pid.I = pid_states.get("I", 0.0)
            self.pid.J_C_prev = pid_states.get("J_C_prev", 0.0)
            self.pid.iteration = pid_states.get("iteration", 0)
