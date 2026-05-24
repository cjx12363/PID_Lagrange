"""Power grid model stub — grid simulation is not needed for PID-Lagrangian work."""
import numpy as np

class PowerGrid:
    def __init__(self, config, env=None, pv_profile=None):
        self.node_num = 34
        self.config = config
        self.env = env
        self.pv_profile = pv_profile

    def reset(self, sim_date, load_data=None, pv_data=None):
        n_nodes = self.node_num
        return np.zeros(n_nodes - 1), np.zeros(n_nodes - 1)

    def step(self, ev_power):
        n_nodes = self.node_num
        return np.zeros(n_nodes - 1), np.zeros(n_nodes - 1), np.ones(n_nodes)
