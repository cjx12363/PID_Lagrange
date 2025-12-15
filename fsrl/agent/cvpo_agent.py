from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb
from torch.distributions import Independent, Normal

from fsrl.agent import OffpolicyAgent
from fsrl.policy import CVPO
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic, SingleCritic


class CVPOAgent(OffpolicyAgent):
    """约束变分策略优化（CVPO）智能体。"""

    name = "CVPOAgent"

    def __init__(
        self,
        env: gym.Env,  # 用于训练智能体的Gym环境
        logger: BaseLogger = BaseLogger(),  # 日志记录器
        # 通用任务参数
        cost_limit: float = 10,  # 任务的成本限制
        device: str = "cpu",  # 用于训练的设备
        thread: int = 4,  # 如果使用"cpu"进行训练
        seed: int = 10,  # 随机种子
        # CVPO参数
        estep_iter_num: int = 1,  # E步的迭代次数
        estep_kl: float = 0.02,  # E步的KL散度阈值
        estep_dual_max: float = 20,  # E步中对偶变量的最大值
        estep_dual_lr: float = 0.02,  # E步中对偶变量的学习率
        sample_act_num: int = 16,  # E步要采样的动作数
        mstep_iter_num: int = 1,  # M步的迭代次数
        mstep_kl_mu: float = 0.005,  # M步的KL散度阈值(均值)
        mstep_kl_std: float = 0.0005,  # M步的KL散度阈值(标准差)
        mstep_dual_max: float = 0.5,  # M步中对偶变量的最大值
        mstep_dual_lr: float = 0.1,  # M步中对偶变量的学习率
        # 其他算法参数
        actor_lr: float = 5e-4,  # 演员网络的学习率
        critic_lr: float = 1e-3,  # 评论家网络的学习率
        gamma: float = 0.98,  # 折扣因子
        n_step: int = 2,  # 计算回报时向前看的步数
        tau: float = 0.05,  # 评论家软更新系数
        hidden_sizes: Tuple[int, ...] = (128, 128),  # 隐藏层大小
        double_critic: bool = False,  # 是否使用两个评论家网络
        conditioned_sigma: bool = True,  # 高斯策略的方差是否以状态为条件
        unbounded: bool = False,  # 是否为演员网络使用无界输出层
        last_layer_scale: bool = False,  # 是否缩放演员网络的最后一层
        deterministic_eval: bool = True,  # 评估期间是否使用确定性策略
        action_scaling: bool = True,  # 是否按最大动作值缩放动作
        action_bound_method: str = "clip",  # 用于动作边界的方法
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

        assert hasattr(
            env.spec, "max_episode_steps"
        ), "Please use an env wrapper to provide 'max_episode_steps' for CVPO"

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
            if double_critic:
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
            else:
                net_c = Net(
                    state_shape,
                    action_shape,
                    hidden_sizes=hidden_sizes,
                    concat=True,
                    device=device
                )
                critics.append(SingleCritic(net_c, device=device).to(device))

        critic_optim = torch.optim.Adam(
            nn.ModuleList(critics).parameters(), lr=critic_lr
        )
        if not conditioned_sigma:
            torch.nn.init.constant_(actor.sigma_param, -0.5)
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

        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = CVPO(
            actor=actor,
            critics=critics,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            logger=logger,
            action_space=env.action_space,
            dist_fn=dist,
            max_episode_steps=env.spec.max_episode_steps,
            cost_limit=cost_limit,
            tau=tau,
            gamma=gamma,
            n_step=n_step,
            estep_iter_num=estep_iter_num,
            estep_kl=estep_kl,
            estep_dual_max=estep_dual_max,
            estep_dual_lr=estep_dual_lr,
            sample_act_num=sample_act_num,  # 用于连续动作空间
            mstep_iter_num=mstep_iter_num,
            mstep_kl_mu=mstep_kl_mu,
            mstep_kl_std=mstep_kl_std,
            mstep_dual_max=mstep_dual_max,
            mstep_dual_lr=mstep_dual_lr,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler
        )
