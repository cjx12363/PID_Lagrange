"""
Cost functions for PID-Lagrangian safe RL.

The primary safety cost is transformer overload: c_t = ReLU((P_base + P_ev) / P_rated - 1).
Also retains the original user-penalty cost as reference.
"""
import math


def transformer_overload_cost(env, total_costs, user_satisfaction_list, *args):
    """Per-step safety cost: fractional transformer overload (0 when safe, >0 when overloaded).

    c_t = max(0, total_power / max_power - 1)  summed over all transformers.
    """
    cost = 0.0
    for tr in env.transformers:
        cost += tr.get_how_overloaded()
    return cost


def transformer_overload_usrpenalty_cost(env, total_costs, user_satisfaction_list, *args):
    """Original cost: transformer overload * 100 + user dissatisfaction penalty."""
    cost = 0
    for tr in env.transformers:
        cost += 100 * tr.get_how_overloaded()
    for score in user_satisfaction_list:
        cost += 100 * math.exp(-10 * score)
    return cost


def ProfitMax_TrPenalty_UserIncentives_safety(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    return reward
