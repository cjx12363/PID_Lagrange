"""Training script v3: Direct cost-penalized reward with PID-tuned lambda.

Architecture:
  augmented_reward = reward - lambda * COST_PENALTY * cost
  PID adjusts lambda per episode based on J_C vs d.
  
This bypasses the Safety Critic entirely -- the cost signal flows directly
through the reward, which the Reward Critic captures reliably.
The PID only needs to find the right lambda to balance profit vs safety.

Key insight: for sparse/small-magnitude costs, Safety Critic learning is
the bottleneck. Direct cost penalty avoids this bottleneck entirely.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, "E:/cjx12363/PID/EV2Gym")
sys.path.insert(0, "E:/cjx12363/PID")
os.chdir("E:/cjx12363/PID/EV2Gym")

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.state import V2G_profit_max_loads
from ev2gym.rl_agent.reward import V2G_profitmaxV2
from ev2gym.rl_agent.cost import transformer_overload_cost

from agent.safe_sac import SafeSAC
from agent.buffer import ReplayBuffer

COST_PENALTY = 20.0  # per-unit overload penalty in reward space (~profit scale)

def make_config(run_type):
    base = {
        "device": "cuda", "gamma": 0.99, "tau": 0.005, "lr": 1e-4,
        "alpha_lr": 3e-4, "batch_size": 256, "cost_scale": 1.0,
        "gradient_steps_per_episode": 50, "total_episodes": 150,
        "eval_every": 25, "eval_episodes": 5,
        "config_file": "ev2gym/example_config_files/PID_Lagrangian.yaml",
        "log_dir": "./logs", "save_dir": "./checkpoints",
        "buffer_capacity": int(1e6),
    }
    if run_type == "pid":
        base["pid"] = {"K_P": 0.3, "K_I": 0.01, "K_D": 0.05, "d": 3.0}
    elif run_type == "lagrangian":
        base["pid"] = {"K_P": 0.0, "K_I": 0.01, "K_D": 0.0, "d": 3.0}
    elif run_type == "unconstrained":
        base["pid"] = {"K_P": 0.0, "K_I": 0.0, "K_D": 0.0, "d": 1e9}
    return base

def evaluate(agent, env, num_episodes=5):
    rewards, costs = [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_r, ep_c, done = 0.0, 0.0, False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_r += reward
            c = info.get("cost", 0.0)
            if c is not None: ep_c += c
        rewards.append(ep_r)
        costs.append(ep_c)
    return np.mean(rewards), np.std(rewards), np.mean(costs), np.std(costs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pid", choices=["pid","lagrangian","unconstrained"])
    args = parser.parse_args()
    config = make_config(args.config)
    run_name = args.config

    log_dir = os.path.join(config["log_dir"], run_name)
    save_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    env = EV2Gym(config_file=config["config_file"],
                 state_function=V2G_profit_max_loads, reward_function=V2G_profitmaxV2,
                 cost_function=transformer_overload_cost,
                 generate_rnd_game=True, verbose=False)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    print(f"Run: {run_name} | Obs: {obs_dim}, Act: {act_dim} | PID: {config['pid']}")

    agent = SafeSAC(obs_dim, act_dim, config)
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=config["buffer_capacity"])

    log = {"run": run_name, "episodes": [], "rewards": [], "costs": [], "lambdas": [], "alphas": [],
           "eval_episodes": [], "eval_rewards": [], "eval_rewards_std": [],
           "eval_costs": [], "eval_costs_std": []}

    start_time = time.time()

    for ep in range(1, config["total_episodes"] + 1):
        obs, _ = env.reset()
        ep_r, ep_c, done = 0.0, 0.0, False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            cost = info.get("cost", 0.0) or 0.0
            # Direct cost penalty: lambda * COST_PENALTY * cost
            aug_reward = reward - agent.lam * COST_PENALTY * cost
            # Safety critic gets cost as-is (for logging; not used in actor loss)
            buffer.add(obs, action, aug_reward, cost * config["cost_scale"],
                       next_obs, float(done))
            ep_r += reward; ep_c += cost; obs = next_obs

        for _ in range(config["gradient_steps_per_episode"]):
            agent.update(buffer)

        agent.update_pid(ep_c)

        log["episodes"].append(ep); log["rewards"].append(ep_r)
        log["costs"].append(ep_c); log["lambdas"].append(agent.lam)
        log["alphas"].append(agent.alpha.item())

        elapsed = time.time() - start_time
        print(f"Ep {ep:4d} | R: {ep_r:8.1f} | C: {ep_c:7.2f} | lam: {agent.lam:6.3f} | t: {elapsed:.0f}s")

        if ep % config["eval_every"] == 0 or ep == config["total_episodes"]:
            mr, sr, mc, sc = evaluate(agent, env, config["eval_episodes"])
            log["eval_episodes"].append(ep); log["eval_rewards"].append(mr)
            log["eval_rewards_std"].append(sr); log["eval_costs"].append(mc)
            log["eval_costs_std"].append(sc)
            print(f"  Eval | R: {mr:8.1f} +- {sr:.1f} | C: {mc:6.2f} +- {sc:.2f}")
            agent.save(os.path.join(save_dir, f"checkpoint_ep{ep}.pt"))
            with open(os.path.join(log_dir, "training_log.json"), "w") as f:
                json.dump(log, f)

    env.close()
    agent.save(os.path.join(save_dir, "final_model.pt"))
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(log, f)
    print(f"\nDone in {time.time()-start_time:.0f}s")

if __name__ == "__main__":
    main()
