"""Quick test script to verify training loop works end-to-end with minimal episodes."""
import os, sys, time, numpy as np, torch

sys.path.insert(0, "E:/cjx12363/PID")
sys.path.insert(0, "E:/cjx12363/PID/EV2Gym")

os.chdir("E:/cjx12363/PID/EV2Gym")

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.state import V2G_profit_max_loads
from ev2gym.rl_agent.reward import V2G_profitmaxV2
from ev2gym.rl_agent.cost import transformer_overload_cost

from agent.safe_sac import SafeSAC
from agent.buffer import ReplayBuffer
from config import CONFIG

config = dict(CONFIG)
config["config_file"] = "ev2gym/example_config_files/PID_Lagrangian.yaml"
config["ev2gym_path"] = "E:/cjx12363/PID/EV2Gym"
config["total_episodes"] = 20
config["gradient_steps_per_episode"] = 200
config["eval_every"] = 10
config["device"] = "cuda"
config["pid"] = {"K_P": 0.10, "K_I": 0.01, "K_D": 0.05, "d": 1.0}
config["cost_scale"] = 1.0  # don't scale since cost is already fractional overload

print("Creating env...")
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
print(f"Obs: {obs_dim}, Act: {act_dim}")
env.close()

agent = SafeSAC(obs_dim, act_dim, config)
buffer = ReplayBuffer(obs_dim, act_dim)

for ep in range(1, config["total_episodes"] + 1):
    env = EV2Gym(
        config_file=config["config_file"],
        state_function=V2G_profit_max_loads,
        reward_function=V2G_profitmaxV2,
        cost_function=transformer_overload_cost,
        generate_rnd_game=True,
        verbose=False,
    )
    obs, _ = env.reset()
    ep_r, ep_c, done = 0.0, 0.0, False
    t0 = time.time()
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        cost = info.get("cost", 0.0) or 0.0
        buffer.add(obs, action, reward, cost * config["cost_scale"], next_obs, float(done))
        ep_r += reward; ep_c += cost; obs = next_obs
    dt = time.time() - t0
    for _ in range(config["gradient_steps_per_episode"]):
        agent.update(buffer)
    agent.update_pid(ep_c)
    print(f"Ep {ep:2d}: R={ep_r:8.2f} C={ep_c:6.2f} lam={agent.lam:.3f} alpha={agent.alpha.item():.4f} time={dt:.1f}s")
    env.close()

print("\nQuick test passed!")
