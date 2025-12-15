# PID 项目 - 模块化结构说明

## 📁 项目结构

本项目已重构为清晰的模块化结构：

```
PID/
├── environments/         # 环境模块
│   ├── models/          # EV2Gym环境核心
│   ├── data/            # 数据处理
│   ├── visuals/         # 可视化工具
│   ├── utilities/       # 工具函数
│   ├── rl_integration/  # RL集成（状态、奖励、成本）
│   └── config/          # 配置文件
│
├── algorithms/          # 算法模块
│   ├── baselines/
│   │   ├── mpc/        # MPC控制算法
│   │   ├── gurobi/     # Gurobi优化模型
│   │   └── heuristics/ # 启发式算法
│   └── fsrl/           # 安全强化学习
│
├── example.py           # 评估示例脚本
├── evaluator.py         # 评估工具
├── train_safe_RL.py     # 安全RL训练脚本
├── train_safe_RL_loads.py
└── train_stable_baselines.py
```

## 🚀 快速开始

### 运行评估示例

```bash
# 激活虚拟环境
conda activate env01

# 运行评估脚本
python example.py
```

### 运行训练

```bash
# 激活虚拟环境
conda activate env01

# 运行安全强化学习训练
python train_safe_RL.py
```

## 📦 模块说明

### environments - 环境模块
包含EV2Gym充电调度环境的所有组件：
- **models**: 环境核心逻辑、EV模型、充电桩、变压器等
- **data**: 数据加载和处理
- **visuals**: 可视化和渲染
- **rl_integration**: RL状态表示、奖励函数、成本函数
- **config**: YAML配置文件

### algorithms - 算法模块
各种充电调度算法：
- **baselines/mpc**: 模型预测控制算法
- **baselines/gurobi**: Gurobi优化求解器（需要gurobi库）
- **baselines/heuristics**: 启发式算法（RoundRobin, ChargeAsFastAsPossible等）
- **fsrl**: 安全强化学习算法（CPO, CVPO, PPO-Lag, SAC-Lag）

### scripts - 脚本模块
- **training**: 训练脚本（train_safe_RL.py等）
- **evaluation**: 评估和示例脚本

## 💡 使用示例

### 导入环境
```python
from environments.models.ev2gym_env import EV2Gym

env = EV2Gym(config_file="environments/config/V2GProfitMax.yaml")
```

### 导入算法
```python
# 启发式算法
from algorithms.baselines.heuristics.heuristics import ChargeAsFastAsPossible

agent = ChargeAsFastAsPossible()
action = agent.get_action(env)
```

```python
# 强化学习算法
from algorithms.fsrl.agent import SACLagAgent

agent = SACLagAgent(env=env, logger=logger, cost_limit=2)
```

## 📝 注意事项

- **MPC/Gurobi模块**: 需要安装gurobi库（需要许可证）
- **FSRL模块**: 需要tianshou、wandb等依赖
- **配置文件**: 路径已更新为`environments/config/`

## 🔄 迁移指南

如果你有旧代码需要更新导入路径：

| 旧路径 | 新路径 |
|--------|--------|
| `from ev2gym.models.*` | `from environments.models.*` |
| `from ev2gym.rl_agent.*` | `from environments.rl_integration.*` |
| `from ev2gym.baselines.*` | `from algorithms.baselines.*` |
| `from fsrl.*` | `from algorithms.fsrl.*` |
| `from cost_functions import *` | `from environments.rl_integration.cost_functions import *` |

## 📊 重构优势

✅ **清晰的职责分离** - 环境、算法、脚本各司其职  
✅ **更好的可维护性** - 模块边界清晰，易于定位代码  
✅ **便于扩展** - 轻松添加新算法或环境变体  
✅ **支持协作开发** - 团队成员可独立开发不同模块
