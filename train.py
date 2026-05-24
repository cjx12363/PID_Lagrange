"""
Main training script for PID-Lagrangian Safe SAC on EV2Gym.

Trains a safe SAC agent with PID-controlled Lagrange multiplier
for transformer overload constraint satisfaction.

Usage: python train.py [--config baseline_name]
  baseline_name: pid (default), lagrangian, unconstrained
"""
import os
import sys
import json
import time
import numpy as np
import torch

# Ensure EV2Gym is on path
sys.path.insert(0, "E:/cjx12363/PID/EV2Gym")
sys.path.insert(0, "E:/cjx12363/PID")
os.chdir("E:/cjx12363/PID/EV2Gym")

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.state import V2G_profit_max_loads
from ev2gym.rl_agent.reward import V2G_profitmaxV2
from ev2gym.rl_agent.cost import transformer_overload_cost

from agent.safe_sac import SafeSAC
from agent.buffer import ReplayBuffer
from config import CONFIG, LAGRANGIAN_CONFIG, UNCONSTRAINED_CONFIG


BASELINES = {
    "pid": CONFIG,
    "lagrangian": LAGRANGIAN_CONFIG,
    "unconstrained": UNCONSTRAINED_CONFIG,
}


def make_env(config):
    return EV2Gym(
        config_file=config["config_file"],
        state_function=V2G_profit_max_loads,
        reward_function=V2G_profitmaxV2,
        cost_function=transformer_overload_cost,
        generate_rnd_game=True,
        verbose=False,
    )


def evaluate(agent, env_config, num_episodes=5):
    rewards, costs = [], []
    for _ in range(num_episodes):
        env = make_env(env_config)
        obs, _ = env.reset()
        ep_r, ep_c, done = 0.0, 0.0, False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_r += reward
            c = info.get("cost", 0.0)
            if c is not None:
                ep_c += c
        rewards.append(ep_r)
        costs.append(ep_c)
        env.close()
    return np.mean(rewards), np.std(rewards), np.mean(costs), np.std(costs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pid", choices=["pid", "lagrangian", "unconstrained"])
    args = parser.parse_args()

    config = BASELINES[args.config]
    run_name = args.config

    log_dir = os.path.join(config["log_dir"], run_name)
    save_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Get dimensions
    env = make_env(config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Run: {run_name} | Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"PID params: {config['pid']}")
    env.close()

    agent = SafeSAC(obs_dim, act_dim, config)
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=config["buffer_capacity"])

    log = {
        "run": run_name,
        "config": {k: str(v) if isinstance(v, dict) else v for k, v in config.items()},
        "episodes": [],
        "rewards": [],
        "costs": [],
        "lambdas": [],
        "alphas": [],
        "eval_episodes": [],
        "eval_rewards": [],
        "eval_rewards_std": [],
        "eval_costs": [],
        "eval_costs_std": [],
    }

    start_time = time.time()

    for ep in range(1, config["total_episodes"] + 1):
        env = make_env(config)
        obs, _ = env.reset()
        ep_r, ep_c, done = 0.0, 0.0, False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            cost = info.get("cost", 0.0)
            if cost is None:
                cost = 0.0
            buffer.add(obs, action, reward, cost * config["cost_scale"], next_obs, float(done))
            ep_r += reward
            ep_c += cost
            obs = next_obs

        # Gradient updates
        for _ in range(config["gradient_steps_per_episode"]):
            agent.update(buffer)

        # PID update
        agent.update_pid(ep_c)

        log["episodes"].append(ep)
        log["rewards"].append(ep_r)
        log["costs"].append(ep_c)
        log["lambdas"].append(agent.lam)
        log["alphas"].append(agent.alpha.item())

        elapsed = time.time() - start_time
        print(f"Ep {ep:4d} | R: {ep_r:8.2f} | C: {ep_c:7.2f} | "
              f"lam: {agent.lam:6.3f} | alpha: {agent.alpha.item():.4f} | "
              f"time: {elapsed:.0f}s")

        if ep % config["eval_every"] == 0 or ep == config["total_episodes"]:
            mean_r, std_r, mean_c, std_c = evaluate(agent, config, config["eval_episodes"])
            log["eval_episodes"].append(ep)
            log["eval_rewards"].append(mean_r)
            log["eval_rewards_std"].append(std_r)
            log["eval_costs"].append(mean_c)
            log["eval_costs_std"].append(std_c)
            print(f"  >>> Eval | R: {mean_r:8.2f} +- {std_r:.2f} | "
                  f"C: {mean_c:6.2f} +- {std_c:.2f}")
            agent.save(os.path.join(save_dir, f"checkpoint_ep{ep}.pt"))
            with open(os.path.join(log_dir, "training_log.json"), "w") as f:
                json.dump(log, f)

        env.close()

    agent.save(os.path.join(save_dir, "final_model.pt"))
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(log, f)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Logs: {log_dir}/training_log.json")
    print(f"Model: {save_dir}/final_model.pt")


if __name__ == "__main__":
    main()
