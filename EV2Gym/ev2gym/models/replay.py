"""Replay serialisation for EV2Gym."""
import pickle

class EvCityReplay:
    def __init__(self, env):
        self.replay_path = getattr(env, 'replay_path', './replay/')
        self.n_cs = env.cs
        self.n_transformers = env.number_of_transformers
        self.max_n_ports = env.number_of_ports_per_cs
        self.scenario = env.scenario
        self.heterogeneous_specs = env.config.get('heterogeneous_ev_specs', True)
        self.simulate_grid = env.simulate_grid
        self.sim_date = env.sim_date
        self.timescale = env.timescale
        self.transformers = env.transformers
        self.charging_stations = env.charging_stations
        self.charge_prices = env.charge_prices
        self.discharge_prices = env.discharge_prices
        self.power_setpoints = env.power_setpoints
        self.EVs = env.EVs
        self.load_data = None
        self.pv_data = None
        self.optimal_stats = None
        self.cs_transformers = getattr(env, 'cs_transformers', [])
        self.grid = getattr(env, 'grid', None)
