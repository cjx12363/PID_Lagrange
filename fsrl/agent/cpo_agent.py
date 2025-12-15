from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.env import BaseVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
from fsrl.policy import CPO
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


class CPOAgent(OnpolicyAgent):
    """CPO（约束策略优化）智能体。"""

    name = "CPOAgent"

    def __init__(
        self,
        env: gym.Env,  # 用于训练和评估智能体的环境
        logger: BaseLogger = BaseLogger(),  # 日志记录器实例
        cost_limit: float = 10,  # 允许的最大约束成本
        device: str = "cpu",  # 用于训练和推理的设备
        thread: int = 4,  # 如果使用"cpu"进行训练
        seed: int = 10,  # 用于可重现性的随机种子
        lr: float = 1e-3,  # 学习率
        hidden_sizes: Tuple[int, ...] = (128, 128),  # 隐藏层大小
        unbounded: bool = False,  # 动作空间是否无界
        last_layer_scale: bool = False,  # 是否缩放策略网络的最后一层输出
        # CPO特定参数
        target_kl: float = 0.01,  # 策略更新的目标KL散度
        backtrack_coeff: float = 0.8,  # 回溯系数
        damping_coeff: float = 0.1,  # 阻尼系数
        max_backtracks: int = 10,  # 最大回溯步数
        optim_critic_iters: int = 10,  # 优化评论家的迭代次数
        l2_reg: float = 0.001,  # L2正则化系数
        gae_lambda: float = 0.95,  # 广义优势估计的lambda参数
        advantage_normalization: bool = True,  # 是否归一化优势
        # 基础策略通用参数
        gamma: float = 0.99,  # 未来奖励和成本的折扣因子
        max_batchsize: int = 99999,  # 计算优势等的最大批次大小
        reward_normalization: bool = False,  # 是否归一化奖励（可能降低最终性能）
        deterministic_eval: bool = True,  # 评估期间是否使用确定性动作
        action_scaling: bool = True,  # 是否缩放动作空间
        action_bound_method: str = "clip",  # 约束动作的方法
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None  # 学习率调度器
    ) -> None:
        super().__init__()

        self.logger = logger
        self.cost_limit = cost_limit

        if not np.isscalar(cost_limit):
            raise RuntimeError(
                "CPO does not support multiple costs. \n \
                Please refer to Page 5 of \
                http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf \
                for related discussions."
            )

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
            ).to(device) for _ in range(2)
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
        # optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        optim = torch.optim.Adam(nn.ModuleList(critic).parameters(), lr=lr)

        # 用Independent(Normal)替换DiagGuassian，它们是等价的
        # 传递*logits以保持与policy.forward一致
        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = CPO(
            actor,
            critic,
            optim,
            dist,  # type: ignore
            logger=logger,
            # CPO特定参数
            target_kl=target_kl,
            backtrack_coeff=backtrack_coeff,
            damping_coeff=damping_coeff,
            max_backtracks=max_backtracks,
            optim_critic_iters=optim_critic_iters,
            l2_reg=l2_reg,
            gae_lambda=gae_lambda,
            advantage_normalization=advantage_normalization,
            cost_limit=cost_limit,
            # 基础策略通用参数
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler
        )

    def learn(
        self,
        train_envs: Union[gym.Env, BaseVectorEnv],
        test_envs: Union[gym.Env, BaseVectorEnv] = None,
        epoch: int = 300,
        episode_per_collect: int = 20,
        step_per_epoch: int = 10000,
        repeat_per_collect: int = 4,
        buffer_size: int = 100000,
        testing_num: int = 2,
        batch_size: int = 99999,
        reward_threshold: float = 450,
        save_interval: int = 4,
        resume: bool = False,
        save_ckpt: bool = True,
        verbose: bool = True,
        show_progress: bool = True
    ) -> None:
        """详情请参见 :meth:`~fsrl.agent.OnpolicyAgent.learn`。"""
        return super().learn(
            train_envs, test_envs, epoch, episode_per_collect, step_per_epoch,
            repeat_per_collect, buffer_size, testing_num, batch_size, reward_threshold,
            save_interval, resume, save_ckpt, verbose, show_progress
        )
