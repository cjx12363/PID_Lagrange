from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
from fsrl.policy import FOCOPS
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


class FOCOPSAgent(OnpolicyAgent):
    """策略空间一阶约束优化（FOCOPS）智能体。"""

    name = "FOCOPSAgent"

    def __init__(
        self,
        env: gym.Env,  # 用于训练和评估智能体的环境
        logger: BaseLogger = BaseLogger(),  # 日志记录器实例
        cost_limit: float = 10,  # 约束阈值
        device: str = "cpu",  # 用于训练和推理的设备
        thread: int = 4,  # 如果使用"cpu"进行训练
        seed: int = 10,  # 用于可重现性的随机种子
        actor_lr: float = 5e-4,  # 演员网络的学习率
        critic_lr: float = 1e-3,  # 评论家网络的学习率
        hidden_sizes: Tuple[int, ...] = (128, 128),  # 隐藏层大小
        unbounded: bool = False,  # 动作空间是否无界
        last_layer_scale: bool = False,  # 是否缩放策略网络的最后一层输出
        # FOCOPS特定参数
        auto_nu: bool = True,  # 是否自动调整nu(成本系数)
        nu: float = 0.01,  # 成本系数
        nu_max: float = 2.0,  # nu的最大值(如果auto_nu为True)
        nu_lr: float = 1e-2,  # nu的学习率(如果auto_nu为True)
        l2_reg: float = 1e-3,  # L2正则化率
        delta: float = 0.02,  # 提前停止KL界
        eta: float = 0.02,  # 指示函数的KL界
        tem_lambda: float = 0.95,  # 逆温度lambda
        gae_lambda: float = 0.95,  # GAE lambda
        max_grad_norm: Optional[float] = 0.5,  # 梯度裁剪的最大梯度范数
        advantage_normalization: bool = True,  # 是否归一化优势
        recompute_advantage: bool = False,  # 是否重新计算优势
        # 基础策略通用参数
        gamma: float = 0.99,  # 未来奖励的折扣因子
        max_batchsize: int = 100000,  # 优化的最大批次大小
        reward_normalization: bool = False,  # 是否归一化奖励(可能降低最终性能)
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

        actor = ActorProb(
            Net(state_shape, hidden_sizes=hidden_sizes, device=device),
            action_shape,
            max_action=max_action,
            unbounded=unbounded,
            device=device
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        critic = [
            Critic(
                Net(state_shape, hidden_sizes=hidden_sizes, device=device),
                device=device
            ).to(device) for _ in range(2)
        ]
        critic_optim = torch.optim.Adam(nn.ModuleList(critic).parameters(), lr=critic_lr)

        torch.nn.init.constant_(actor.sigma_param, -0.5)
        actor_critic = ActorCritic(actor, critic)
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

        # 用Independent(Normal)替换DiagGuassian，它们是等价的
        # 传递*logits以保持与policy.forward一致
        def dist(*logits):
            return Independent(Normal(*logits), 1)

        if auto_nu:
            nu = torch.zeros(1, requires_grad=False, device=device)
            nu = (nu_max, nu_lr, nu)

        self.policy = FOCOPS(
            actor,
            critic,
            actor_optim,
            critic_optim,
            dist,
            logger=logger,
            cost_limit=cost_limit,
            nu=nu,
            l2_reg=l2_reg,
            delta=delta,
            eta=eta,
            tem_lambda=tem_lambda,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler,
        )
