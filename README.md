# PID-Lagrangian Safe RL for EV Charging Station Scheduling

基于 PID-拉格朗日方法的深度强化学习，用于电动汽车充电站的安全调度。

## Overview

将 **PID-Lagrangian** 方法（Stooke et al., ICML 2020）应用于电动汽车充电站的能量调度问题。在追求充电利润最大化的同时，通过 PID 控制器动态调节拉格朗日乘子，确保变压器不过载（安全约束）。

## Architecture

\\\
augmented_reward = profit_reward - lambda * COST_PENALTY * overload_cost

PID controller adjusts lambda per episode:
  lambda = K_P * error + K_I * integral + K_D * derivative
  error  = J_C - d   (actual cost vs safety threshold)
\\\

核心思路：绕过 Safety Critic 的学习瓶颈，直接将成本惩罚注入奖励信号，由 Reward Critic 可靠捕获。PID 只需找到平衡利润与安全的正确 lambda。

## Project Structure

\\\
PID_Lagrange/
├── agent/                  # PID-Lagrangian Safe SAC 核心
│   ├── pid_lagrangian.py   # PID 乘子控制器
│   ├── safe_sac.py         # Safe SAC 智能体
│   ├── networks.py         # Actor/Critic 网络
│   └── buffer.py           # 经验回放缓冲区
├── EV2Gym/                 # EV 充电站仿真环境
│   ├── ev2gym/             # 环境核心（models, rl_agent, utilities, visuals）
│   ├── evaluator.py        # 模型评估
│   ├── example.py          # 环境示例
│   └── train_stable_baselines.py  # Baselines 训练脚本
├── config.py               # 训练配置（PID / Lagrangian / Unconstrained）
├── train.py                # 训练入口（v3: 直接成本惩罚 + PID lambda 调节）
├── plot_results.py         # 结果可视化（reward, cost, lambda, Pareto）
└── .gitignore
\\\

## Quick Start

\\\ash
# 安装依赖
pip install torch numpy matplotlib pyyaml gymnasium

# 训练 PID-Lagrangian 模型
python train.py --method pid

# 对比训练 Lagrangian 基线
python train.py --method lagrangian

# 对比训练无约束 SAC
python train.py --method unconstrained

# 绘制结果
python plot_results.py
\\\

## Key Configs

\config.py\ 中预设了三种配置：

| Method | K_P | K_I | K_D | d |
|--------|-----|-----|-----|---|
| PID-Lagrangian | 0.05 | 0.005 | 0.02 | 5.0 |
| Lagrangian | 0.5 | 0.005 | 0.0 | 5.0 |
| Unconstrained | 0.0 | 0.0 | 0.0 | 1e9 |

## Reference

- **PID-Lagrangian**: Stooke et al. "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods." ICML 2020.
- **SAC**: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL." ICML 2018.
- **EV2Gym**: EV charging station simulation environment.

## License

MIT
