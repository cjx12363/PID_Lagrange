"""
PID-Lagrangian multiplier updater.

Based on Stooke et al. (ICML 2020), Algorithm 2.
Uses proportional, integral, and derivative terms to stabilize
the Lagrange multiplier for constrained RL.
"""
import numpy as np


class PIDLagrangianUpdater:
    """PID controller for the Lagrange multiplier lambda.

    Args:
        K_P: proportional gain
        K_I: integral gain
        K_D: derivative gain (only active when cost is increasing)
        d: safety threshold (maximum allowed episodic cost)
    """

    def __init__(self, K_P=0.10, K_I=0.01, K_D=0.05, d=25.0):
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.d = d
        self.I = 0.0          # integral accumulator
        self.J_C_prev = 0.0   # previous episodic cost
        self.iteration = 0

    def update(self, J_C):
        """Update and return the Lagrange multiplier lambda.

        Args:
            J_C: non-discounted episodic cost for the most recent episode.

        Returns:
            lambda: non-negative Lagrange multiplier.
        """
        Delta = J_C - self.d                         # error
        partial = max(J_C - self.J_C_prev, 0.0)      # derivative (only on increase)
        self.I = max(self.I + Delta, 0.0)            # integral (clipped >= 0)
        lam = self.K_P * Delta + self.K_I * self.I + self.K_D * partial
        lam = max(lam, 0.0)                          # non-negative projection
        self.J_C_prev = J_C
        self.iteration += 1
        return lam

    def reset(self):
        """Reset internal state."""
        self.I = 0.0
        self.J_C_prev = 0.0
        self.iteration = 0
