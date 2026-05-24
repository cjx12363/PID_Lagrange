"""
Plotting script for PID-Lagrangian training results.

Generates:
  1. Reward curves (smoothed) for PID vs Lagrangian vs Unconstrained
  2. Cost curves
  3. Lambda convergence curves
  4. Pareto frontier (reward vs cost scatter)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Data paths
BASE = "E:/cjx12363/PID/EV2Gym"
RUNS = {
    "PID-Lagrangian (ours)": "logs/pid/training_log.json",
    "Lagrangian SAC": "logs/lagrangian/training_log.json",
    "Unconstrained SAC": "logs/unconstrained/training_log.json",
}

OUT_DIR = "E:/cjx12363/PID/results"
os.makedirs(OUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.figsize": (10, 5),
})

def smooth(data, window=5):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode="valid")

def load_data(path):
    with open(os.path.join(BASE, path)) as f:
        return json.load(f)

datas = {name: load_data(path) for name, path in RUNS.items()}

# ---------- Figure 1: Reward curves ----------
fig, ax = plt.subplots(figsize=(10, 5))
for name, d in datas.items():
    eps = np.array(d["episodes"])
    rewards = np.array(d["rewards"])
    if len(rewards) > 5:
        r_smooth = smooth(rewards, 5)
        e_smooth = eps[4:]
    else:
        r_smooth, e_smooth = rewards, eps
    ax.plot(e_smooth, r_smooth, label=name, linewidth=1.5, alpha=0.9)

ax.set_xlabel("Episode")
ax.set_ylabel("Episode Reward (smoothed)")
ax.set_title("Training Reward Curves")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "reward_curves.png"))
print("Saved reward_curves.png")

# ---------- Figure 2: Cost curves ----------
fig, ax = plt.subplots(figsize=(10, 5))
for name, d in datas.items():
    eps = np.array(d["episodes"])
    costs = np.array(d["costs"])
    if len(costs) > 5:
        c_smooth = smooth(costs, 5)
        e_smooth = eps[4:]
    else:
        c_smooth, e_smooth = costs, eps
    ax.plot(e_smooth, c_smooth, label=name, linewidth=1.5, alpha=0.9)

# Add safety threshold line
d_threshold = 1.0
ax.axhline(y=d_threshold, color="red", linestyle="--", alpha=0.6, label=f"Safety threshold (d={d_threshold})")
ax.set_xlabel("Episode")
ax.set_ylabel("Episode Cost (smoothed)")
ax.set_title("Training Cost (Transformer Overload) Curves")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "cost_curves.png"))
print("Saved cost_curves.png")

# ---------- Figure 3: Lambda convergence ----------
fig, ax = plt.subplots(figsize=(10, 5))
for name in ["PID-Lagrangian (ours)", "Lagrangian SAC"]:
    if name in datas:
        d = datas[name]
        eps = np.array(d["episodes"])
        lambdas = np.array(d["lambdas"])
        ax.plot(eps, lambdas, label=name, linewidth=1.5, alpha=0.9)

ax.set_xlabel("Episode")
ax.set_ylabel("Lambda (Lagrange Multiplier)")
ax.set_title("PID vs Lagrangian Lambda Convergence")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lambda_curves.png"))
print("Saved lambda_curves.png")

# ---------- Figure 4: Pareto frontier ----------
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"PID-Lagrangian (ours)": "#2ecc71", "Lagrangian SAC": "#3498db", "Unconstrained SAC": "#e74c3c"}
markers = {"PID-Lagrangian (ours)": "o", "Lagrangian SAC": "s", "Unconstrained SAC": "^"}

for name, d in datas.items():
    if "eval_rewards" in d and len(d["eval_rewards"]) > 0:
        eval_eps = np.array(d["eval_episodes"])
        eval_r = np.array(d["eval_rewards"])
        eval_r_std = np.array(d["eval_rewards_std"])
        eval_c = np.array(d["eval_costs"])
        eval_c_std = np.array(d["eval_costs_std"])
        # Plot last eval point
        ax.errorbar(eval_c[-1], eval_r[-1],
                    xerr=eval_c_std[-1], yerr=eval_r_std[-1],
                    fmt=markers[name], color=colors[name],
                    label=name, markersize=10, capsize=4, alpha=0.9)

ax.set_xlabel("Episode Cost (transformer overload)")
ax.set_ylabel("Episode Reward")
ax.set_title("Pareto Frontier: Reward vs Cost (final eval)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=d_threshold, color="red", linestyle="--", alpha=0.5, label=f"Safety threshold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pareto_frontier.png"))
print("Saved pareto_frontier.png")

# ---------- Figure 5: Combined summary ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Reward
ax = axes[0]
for name, d in datas.items():
    eps = np.array(d["episodes"])
    rewards = np.array(d["rewards"])
    if len(rewards) > 5:
        r_smooth = smooth(rewards, 5)
        e_smooth = eps[4:]
    else:
        r_smooth, e_smooth = rewards, eps
    ax.plot(e_smooth, r_smooth, label=name, linewidth=1.5)
ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Reward")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Cost
ax = axes[1]
for name, d in datas.items():
    eps = np.array(d["episodes"])
    costs = np.array(d["costs"])
    if len(costs) > 5:
        c_smooth = smooth(costs, 5)
        e_smooth = eps[4:]
    else:
        c_smooth, e_smooth = costs, eps
    ax.plot(e_smooth, c_smooth, label=name, linewidth=1.5)
ax.axhline(y=d_threshold, color="red", linestyle="--", alpha=0.6)
ax.set_xlabel("Episode"); ax.set_ylabel("Cost"); ax.set_title("Cost")
ax.grid(True, alpha=0.3)

# Lambda
ax = axes[2]
for name in ["PID-Lagrangian (ours)", "Lagrangian SAC"]:
    if name in datas:
        d = datas[name]
        eps = np.array(d["episodes"])
        lambdas = np.array(d["lambdas"])
        ax.plot(eps, lambdas, label=name, linewidth=1.5)
ax.set_xlabel("Episode"); ax.set_ylabel("Lambda"); ax.set_title("Lambda")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("PID-Lagrangian Safe RL on EV2Gym", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "summary.png"))
print("Saved summary.png")

print(f"\nAll plots saved to {OUT_DIR}/")
