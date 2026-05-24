"""Fixed training script with:
1. Cost added directly to reward as penalty (scale=100) so actor learns to avoid overload
2. Realistic d based on environment dynamics (d=5, achievable target)
3. Env reuse + reduced gradient steps (kept from optimization)
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
from config import CONFIG, LAGRANGIAN_CONFIG, UNCONSTRAINED_CONFIG

COST_PENALTY = 100.0  # multiplier: cost -> reward penalty, makes safety signal comparable to profit signal

BASELINES = {
    "pid": CONFIG,
    "lagrangian": LAGRANGIAN_CONFIG,
    "unconstrained": UNCONSTRAINED_CONFIG,
}

# Fix PID params for realistic d
for cfg in [BASELINES["pid"], BASELINES["lagrangian"]]:
    cfg["pid"]["d"] = 5.0
    cfg["pid"]["K_P"] = 0.05
    cfg["pid"]["K_I"] = 0.005
    cfg["pid"]["K_D"] = 0.02

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
            if c is not None:
                ep_c += c
        rewards.append(ep_r)
        costs.append(ep_c)
    return np.mean(rewards), np.std(rewards), np.mean(costs), np.std(costs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pid",
                        choices=["pid", "lagrangian", "unconstrained"])
    args = parser.parse_args()

    config = BASELINES[args.config]
    run_name = args.config

    log_dir = os.path.join(config["log_dir"], run_name)
    save_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    env = EV2Gym(
        config_file=config["config_file"],
        state_function=V2G_profit_max_loads,
        reward_function=V2G_profitmaxV2,
        cost_function=transformer_overload_cost,
        generate_rnd_game=True,
        verbose=False,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Run: {run_name} | Obs: {obs_dim}, Act: {act_dim}")
    print(f"PID: {config['pid']} | Gradient steps/ep: {config['gradient_steps_per_episode']}")

    agent = SafeSAC(obs_dim, act_dim, config)
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=config["buffer_capacity"])

    log = {
        "run": run_name,
        "episodes": [], "rewards": [], "costs": [], "lambdas": [], "alphas": [],
        "eval_episodes": [], "eval_rewards": [], "eval_rewards_std": [],
        "eval_costs": [], "eval_costs_std": [],
    }

    start_time = time.time()

    for ep in range(1, config["total_episodes"] + 1):
        obs, _ = env.reset()
        ep_r, ep_c, done = 0.0, 0.0, False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            cost = info.get("cost", 0.0)
            if cost is None:
                cost = 0.0
            # Safety penalty: subtract cost from reward so actor directly learns to avoid overload
            augmented_reward = reward - COST_PENALTY * cost
            buffer.add(obs, action, augmented_reward, cost * config["cost_scale"],
                       next_obs, float(done))
            ep_r += reward
            ep_c += cost
            obs = next_obs

        for _ in range(config["gradient_steps_per_episode"]):
            agent.update(buffer)

        agent.update_pid(ep_c)

        log["episodes"].append(ep)
        log["rewards"].append(ep_r)
        log["costs"].append(ep_c)
        log["lambdas"].append(agent.lam)
        log["alphas"].append(agent.alpha.item())

        elapsed = time.time() - start_time
        print(f"Ep {ep:4d} | R: {ep_r:8.2f} | C: {ep_c:7.2f} | "
              f"lam: {agent.lam:6.3f} | alpha: {agent.alpha.item():.4f} | "
              f"t: {elapsed:.0f}s")

        if ep % config["eval_every"] == 0 or ep == config["total_episodes"]:
            mean_r, std_r, mean_c, std_c = evaluate(agent, env, config["eval_episodes"])
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


if __name__ == "__main__":
    main()
