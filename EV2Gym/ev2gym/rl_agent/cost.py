"""Cost functions for PID-Lagrangian safe RL."""
import math


def squared_overload_cost(env, total_costs, user_satisfaction_list, *args):
    """Per-step safety cost: squared fractional overload.
    
    c_t = sum(ReLU(total_power / max_power - 1)^2) * scale
    Squaring makes the Safety Critic loss surface smoother and more informative.
    """
    cost = 0.0
    for tr in env.transformers:
        overload = tr.get_how_overloaded()  # ReLU(total_power / max_power - 1)
        if overload > 0:
            cost += overload   # NOT squared -- keep it simple for Safety Critic
    return cost


def transformer_overload_cost(env, total_costs, user_satisfaction_list, *args):
    """Per-step safety cost: fractional transformer overload."""
    cost = 0.0
    for tr in env.transformers:
        cost += tr.get_how_overloaded()
    return cost


def transformer_utilization_cost(env, total_costs, user_satisfaction_list, *args):
    """Continuous transformer utilization ratio (every step has signal)."""
    cost = 0.0
    for tr in env.transformers:
        step = max(env.current_step - 1, 0)
        if step < len(tr.max_power):
            max_p = tr.max_power[step]
            if max_p > 0:
                cost += max(0.0, tr.current_power) / max_p
    return cost


def ProfitMax_TrPenalty_UserIncentives_safety(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    return reward
