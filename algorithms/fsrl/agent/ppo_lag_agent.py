from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
from fsrl.policy import PPOLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


class PPOLagAgent(OnpolicyAgent):
    """带PID拉格朗日的近端策略优化（PPO）智能体。"""

    name = "PPOLagAgent"

    def __init__(
        self,
        env: gym.Env,  # 用于训练和评估智能体的环境
        logger: BaseLogger = BaseLogger(),  # 日志记录器实例
        cost_limit: float = 10,  # 拉格朗日优化的约束限制
        device: str = "cpu",  # 用于训练和推理的设备
        thread: int = 4,  # 如果使用"cpu"进行训练
        seed: int = 10,  # 用于可重现性的随机种子
        lr: float = 5e-4,  # 学习率
        hidden_sizes: Tuple[int, ...] = (128, 128),  # 隐藏层大小
        unbounded: bool = False,  # 动作空间是否无界
        last_layer_scale: bool = False,  # 是否缩放策略网络的最后一层输出
        # PPO特定参数
        target_kl: float = 0.02,  # PPO更新的目标KL散度
        vf_coef: float = 0.25,  # 损失函数的价值函数系数
        max_grad_norm: Optional[float] = None,  # 梯度裁剪的最大梯度范数
        gae_lambda: float = 0.95,  # GAE参数
        eps_clip: float = 0.2,  # PPO裁剪参数
        dual_clip: Optional[float] = None,  # PPO双裁剪参数
        value_clip: bool = False,  # 是否裁剪价值函数更新
        advantage_normalization: bool = True,  # 是否归一化优势
        recompute_advantage: bool = False,  # 是否重新计算优势
        # 拉格朗日特定参数
        use_lagrangian: bool = True,  # 是否使用拉格朗日约束优化
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),  # PID系数
        rescaling: bool = True,  # 是否使用重缩放技巧
        # 基础策略通用参数
        gamma: float = 0.99,  # 未来奖励的折扣因子
        max_batchsize: int = 99999,  # 计算GAE时的最大批次大小
        reward_normalization: bool = False,  # 是否归一化奖励(可能降低最终性能)
        deterministic_eval: bool = True,  # 评估时是否使用确定性动作
        action_scaling: bool = True,  # 是否根据动作空间缩放动作
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
            net, action_shape, max_action=max_action, unbounded=unbounded, device=device
        ).to(device)
        critic = [
            Critic(
                Net(state_shape, hidden_sizes=hidden_sizes, device=device),
                device=device
            ).to(device) for _ in range(1 + cost_dim)
        ]

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
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        # 用Independent(Normal)替换DiagGuassian，它们是等价的
        # 传递*logits以保持与policy.forward一致
        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = PPOLagrangian(
            actor,
            critic,
            optim,
            dist,
            logger=logger,
            # PPO特定参数
            target_kl=target_kl,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            eps_clip=eps_clip,
            dual_clip=dual_clip,
            value_clip=value_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            # 拉格朗日特定参数
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            # 基础策略通用参数
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
