from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb

from fsrl.agent import OffpolicyAgent
from fsrl.policy import SACLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic


class SACLagAgent(OffpolicyAgent):
    """带PID拉格朗日的软演员-评论家（SAC）智能体。"""

    name = "SACLagAgent"

    def __init__(
        self,
        env: gym.Env,  # 用于训练和评估智能体的环境
        logger: BaseLogger = BaseLogger(),  # 日志记录器实例
        cost_limit: float = 10,  # 拉格朗日优化的约束限制
        # 通用任务参数
        device: str = "cpu",  # 用于训练和推理的设备
        thread: int = 4,  # 如果使用"cpu"进行训练
        seed: int = 10,  # 用于可重现性的随机种子
        # 算法参数
        actor_lr: float = 5e-4,  # 演员网络的学习率
        critic_lr: float = 1e-3,  # 评论家网络的学习率
        hidden_sizes: Tuple[int, ...] = (128, 128),  # 隐藏层大小
        auto_alpha: bool = True,  # 是否自动调整alpha（温度）
        alpha_lr: float = 3e-4,  # 如果auto_alpha为True，alpha的学习率
        alpha: float = 0.002,  # 熔规则化的初始温度
        tau: float = 0.05,  # 目标网络软更新的目标平滑系数
        n_step: int = 2,  # 多步学习的步数
        # 拉格朗日特定参数
        use_lagrangian: bool = True,  # 是否使用拉格朗日约束优化
        lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1),  # PID系数
        rescaling: bool = True,  # 是否使用重缩放技巧
        # 基础策略通用参数
        gamma: float = 0.99,  # 未来奖励的折扣因子
        conditioned_sigma: bool = True,  # 高斯策略的方差是否以状态为条件
        unbounded: bool = True,  # 动作空间是否无界
        last_layer_scale: bool = False,  # 是否缩放策略网络的最后一层输出
        deterministic_eval: bool = False,  # 评估时是否使用确定性动作选择
        action_scaling: bool = True,  # 是否根据动作空间边界缩放动作
        action_bound_method: str = "clip",  # 处理超界动作的方法
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None  # 学习率调度器
    ) -> None:
        super().__init__()

        self.logger = logger
        self.cost_limit = cost_limit

        if np.isscalar(cost_limit):
            cost_dim = 1
        else:
            cost_dim = len(cost_limit)

        # 设置种子和计算
        seed_all(seed)
        torch.set_num_threads(thread)

        # 模型
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]

        net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = ActorProb(
            net,
            action_shape,
            max_action=max_action,
            device=device,
            conditioned_sigma=conditioned_sigma,
            unbounded=unbounded
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        critics = []
        for _ in range(1 + cost_dim):
            net1 = Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            )
            net2 = Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            )
            critics.append(DoubleCritic(net1, net2, device=device).to(device))

        critic_optim = torch.optim.Adam(
            nn.ModuleList(critics).parameters(), lr=critic_lr
        )

        actor_critic = ActorCritic(actor, critics)
        # 正交初始化
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        if last_layer_scale:
            # 进行最后的策略层缩放，这将使初始动作具有（接近于）0均值和标准差，
            # 有助于提升性能，详见
            # https://arxiv.org/abs/2006.05990，图24
            for m in actor.mu.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        if auto_alpha:
            target_entropy = -np.prod(env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)

        self.policy = SACLagrangian(
            actor=actor,
            critics=critics,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            logger=logger,
            alpha=alpha,
            tau=tau,
            gamma=gamma,
            exploration_noise=None,
            n_step=n_step,
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            reward_normalization=False,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler
        )
