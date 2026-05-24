import numpy as np

class Transformer:
    """Models a distribution transformer serving one or more EV charging stations.

    Attributes:
        id: transformer index.
        cs_ids: list of charging station ids connected to this transformer.
        max_power: array of shape (simulation_length,) — rated power limit per step (kW).
        inflexible_load: array of background (residential/office) load per step (kW).
        solar_power: array of local PV generation per step (kW).
        simulation_length: total number of steps in the simulation.
        env: back-reference to the EV2Gym environment.
    """

    def __init__(self, id, env, cs_ids, max_power, inflexible_load, solar_power, simulation_length):
        self.id = id
        self.env = env
        self.cs_ids = list(cs_ids) if cs_ids is not None else []
        self.simulation_length = simulation_length

        # Broadcast scalar max_power to per-step array if needed
        if isinstance(max_power, (int, float, np.integer, np.floating)):
            self.max_power = np.full(simulation_length, float(max_power))
        else:
            self.max_power = np.asarray(max_power, dtype=np.float64)
        self.inflexible_load = np.asarray(inflexible_load, dtype=np.float64)
        self.solar_power = np.asarray(solar_power, dtype=np.float64)

        self.current_amps = 0.0
        self.current_power = 0.0

    def reset(self, step=0):
        self.current_amps = 0.0
        self.current_power = 0.0

    def step(self, current_amps, current_power):
        """Called each env step to accumulate total load on this transformer."""
        self.current_amps += current_amps
        self.current_power += current_power

    def get_how_overloaded(self):
        """Fractional overload: ReLU(total_power / max_power - 1)."""
        step = max(self.env.current_step - 1, 0)
        if step >= self.simulation_length:
            step = self.simulation_length - 1
        max_p = self.max_power[step]
        if max_p <= 0:
            return 0.0
        return max(0.0, self.current_power / max_p - 1.0)

    def get_load_pv_forecast(self, step, horizon):
        """Return (net_load, pv) arrays of length `horizon`."""
        end = min(step + horizon, self.simulation_length)
        load = self.inflexible_load[step:end]
        pv = self.solar_power[step:end]
        if len(load) < horizon:
            load = np.pad(load, (0, horizon - len(load)))
            pv = np.pad(pv, (0, horizon - len(pv)))
        return load, pv

    def get_power_limits(self, step, horizon):
        """Return max_power array of length `horizon`."""
        end = min(step + horizon, self.simulation_length)
        limits = self.max_power[step:end]
        if len(limits) < horizon:
            limits = np.pad(limits, (0, horizon - len(limits)), constant_values=limits[-1] if len(limits) > 0 else 0)
        return limits
