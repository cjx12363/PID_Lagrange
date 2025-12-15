from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

from fsrl.agent import OffpolicyAgent
from fsrl.policy import DDPGLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


class DDPGLagAgent(OffpolicyAgent):
    """带PID拉格朗日的深度确定性策略梯度（DDPG）智能体。"""

    name = "DDPGLagAgent"

    def __init__(
        self,
        env: gym.Env,  # 用于训练和评估智能体的环境
        logger: BaseLogger = BaseLogger(),  # 日志记录器实例
        # 通用任务参数
        cost_limit: float = 10,  # 允许的最大约束成本
        device: str = "cpu",  # 用于训练和推理的设备
        thread: int = 4,  # 如果使用"cpu"进行训练
        seed: int = 10,  # 用于可重现性的随机种子
        # 算法参数
        actor_lr: float = 1e-4,  # 演员网络的学习率
        critic_lr: float = 1e-3,  # 评论家网络的学习率
        hidden_sizes: Tuple[int, ...] = (128, 128),  # 隐藏层大小
        tau: float = 0.005,  # 更新目标网络的软更新系数
        exploration_noise: float = 0.1,  # 用于探索的噪声
        n_step: int = 3,  # 多步自举目标的步数
        # 拉格朗日特定参数
        use_lagrangian: bool = True,  # 是否使用拉格朗日约束优化
        lagrangian_pid: Tuple[float, ...] = (0.5, 0.001, 0.1),  # PID系数
        rescaling: bool = True,  # 是否使用重缩放技巧
        # 基础策略通用参数
        gamma: float = 0.99,  # 未来奖励的折扣因子
        deterministic_eval: bool = True,  # 评估时是否使用确定性动作选择
        action_scaling: bool = True,  # 是否根据动作空间边界缩放动作
        action_bound_method: str = "clip",  # 处理超界动作的方法
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None  # 学习率调度器
    ) -> None:
        super().__init__()

        self.logger = logger
        self.cost_limit = cost_limit

        # 设置种子和计算
        seed_all(seed)
        torch.set_num_threads(thread)

        # 模型
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]

        net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = Actor(net, action_shape, max_action=max_action, device=device).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        if np.isscalar(cost_limit):
            cost_dim = 1
        else:
            cost_dim = len(cost_limit)

        nets = [
            Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            ) for i in range(cost_dim + 1)
        ]
        critic = [Critic(n, device=device).to(device) for n in nets]
        critic_optim = torch.optim.Adam(nn.ModuleList(critic).parameters(), lr=critic_lr)

        actor_critic = ActorCritic(actor, critic)
        # 正交初始化
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.policy = DDPGLagrangian(
            actor=actor,
            critics=critic,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            logger=logger,
            tau=tau,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            n_step=n_step,
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            gamma=gamma,
            reward_normalization=False,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler
        )
