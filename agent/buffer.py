"""Simple replay buffer for SafeSAC."""
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=int(1e6)):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards_buf = np.zeros(capacity, dtype=np.float32)
        self.costs_buf = np.zeros(capacity, dtype=np.float32)
        self.dones_buf = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, cost, next_obs, done):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.actions_buf[idx] = action
        self.rewards_buf[idx] = reward
        self.costs_buf[idx] = cost
        self.next_obs_buf[idx] = next_obs
        self.dones_buf[idx] = done
        self.ptr = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs_buf[idxs],
            "actions": self.actions_buf[idxs],
            "rewards": self.rewards_buf[idxs],
            "costs": self.costs_buf[idxs],
            "next_obs": self.next_obs_buf[idxs],
            "dones": self.dones_buf[idxs],
        }

    def __len__(self):
        return self.size
