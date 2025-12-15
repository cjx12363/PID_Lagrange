from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.data import FastCollector
from fsrl.policy import BasePolicy
from fsrl.trainer import OffpolicyTrainer, OnpolicyTrainer
from fsrl.utils import BaseLogger


class BaseAgent(ABC):
    """默认智能体的基类。

    一个智能体类应该包含以下部分：

    * :meth:`~fsrl.agent.BaseAgent.__init__`: 初始化智能体，包括策略、
        网络、优化器等；
    * :meth:`~fsrl.agent.BaseAgent.learn`: 根据给定的学习参数开始训练；
    * :meth:`~fsrl.agent.BaseAgent.evaluate`: 在多个回合中评估智能体；
    * :attr:`~fsrl.agent.BaseAgent.state_dict`: 智能体状态字典，可以
        保存为检查点；

    使用示例：::

        # 初始化CVPO智能体
        agent = CVPOAgent(env, other_algo_params) # 训练多个epoch
        agent.learn(training_envs, other_training_params)

        # 训练完成后测试 agent.eval(testing_envs)

        # 使用智能体的state_dict测试 agent.eval(testing_envs, agent.state_dict)

    所有智能体类都必须继承 :class:`~fsrl.agent.BaseAgent`。
    """

    name = "BaseAgent"

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        self.policy: BasePolicy
        self.task = None
        self.logger = BaseLogger()
        self.cost_limit = 0

    @abstractmethod
    def learn(self, *args, **kwargs) -> None:
        """在一组训练环境上训练策略。"""
        raise NotImplementedError

    def evaluate(
        self,
        test_envs: Union[gym.Env, BaseVectorEnv],
        state_dict: Optional[dict] = None,
        eval_episodes: int = 10,
        render: bool = False,
        train_mode: bool = False
    ) -> Tuple[float, float, float]:
        """在一组测试环境上评估策略。

        :param Union[gym.Env, BaseVectorEnv] test_envs: 用于评估策略的单个环境或
            向量化环境。
        :param Optional[dict] state_dict: 包含要评估的智能体状态参数的可选字典，
             默认为None
        :param int eval_episodes: 要评估的回合数，默认为10
        :param bool render: 评估期间是否渲染环境，默认为False
        :param bool train_mode: 评估期间是否将策略设置为训练模式，
            默认为False
        :return Tuple: 评估期间获得的奖励、回合长度和约束成本。
        """
        if state_dict is not None:
            self.policy.load_state_dict(state_dict)
        if train_mode:
            self.policy.train()
        else:
            self.policy.eval()

        eval_collector = FastCollector(self.policy, test_envs)
        result = eval_collector.collect(n_episode=eval_episodes, render=render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        # term, trun = result["terminated"], result["truncated"] print(f"终止:
        # {term}, 截断: {trun}") print(f"评估奖励: {rews.mean()}, 成本: {cost},
        # 长度: {lens.mean()}")
        return rews, lens, cost

    @property
    def state_dict(self):
        """返回策略的state_dict。"""
        return self.policy.state_dict()


class OffpolicyAgent(BaseAgent):
    """离线策略智能体的基类。

    :meth:`~srl.agent.OffpolicyAgent.learn`: 函数经过定制以与
    离线策略训练器配合工作。更多详细信息请参见 :class:`~fsrl.agent.BaseAgent`。
    """

    name = "OffpolicyAgent"

    def __init__(self) -> None:
        super().__init__()

    def learn(
        self,
        train_envs: Union[gym.Env, BaseVectorEnv],
        test_envs: Union[gym.Env, BaseVectorEnv] = None,
        epoch: int = 300,
        episode_per_collect: int = 5,
        step_per_epoch: int = 3000,
        update_per_step: float = 0.1,
        buffer_size: int = 100000,
        testing_num: int = 2,
        batch_size: int = 256,
        reward_threshold: float = 450,
        save_interval: int = 4,
        resume: bool = False,  # TODO
        save_ckpt: bool = True,
        verbose: bool = True,
        show_progress: bool = True
    ) -> None:
        """在一组训练环境上训练策略。

        :param Union[gym.Env, BaseVectorEnv] train_envs: 用于训练策略的单个环境或
            向量化环境。
        :param Union[gym.Env, BaseVectorEnv] test_envs: 用于评估策略的单个环境或
            向量化环境，默认为None。
        :param int epoch: 训练epoch数，默认为300。
        :param int episode_per_collect: 每次策略更新前收集的回合数，
            默认为5。
        :param int step_per_epoch: 每个epoch的环境步数，默认为
            3000。
        :param float update_per_step: 策略更新与环境步数的比率，\
            默认为0.1。
        :param int buffer_size: 重放缓冲区的最大大小，默认为
            100000。
        :param int testing_num: 用于评估的回合数，默认为
            2。
        :param int batch_size: 每次策略更新的批次大小，默认为256。
        :param float reward_threshold: 提前停止的奖励阈值，\
            默认为450。
        :param int save_interval: 保存策略模型的间隔（以epoch为单位），
            默认为4。
        :param bool resume: 是否从上一个检查点恢复训练，默认为
            False。
        :param bool save_ckpt: 是否保存策略模型，默认为True。
        :param bool verbose: 是否在训练期间打印进度信息，
            默认为True。
        :param bool show_progress: 是否显示tqdm训练进度条，
            默认为True
        """
        assert self.policy is not None, "策略未初始化"
        # 将策略设置为训练模式
        self.policy.train()
        # 收集器
        if isinstance(train_envs, gym.Env):
            buffer = ReplayBuffer(buffer_size)
        else:
            buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        train_collector = FastCollector(
            self.policy,
            train_envs,
            buffer,
            exploration_noise=True,
        )

        test_collector = FastCollector(
            self.policy, test_envs
        ) if test_envs is not None else None

        def stop_fn(reward, cost):
            return reward > reward_threshold and cost < self.cost_limit

        def checkpoint_fn():
            return {"model": self.state_dict}

        if save_ckpt:
            self.logger.setup_checkpoint_fn(checkpoint_fn)

        # 训练器
        trainer = OffpolicyTrainer(
            policy=self.policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epoch,
            batch_size=batch_size,
            cost_limit=self.cost_limit,
            step_per_epoch=step_per_epoch,
            update_per_step=update_per_step,
            episode_per_test=testing_num,
            episode_per_collect=episode_per_collect,
            stop_fn=stop_fn,
            logger=self.logger,
            resume_from_log=resume,
            save_model_interval=save_interval,
            verbose=verbose,
            show_progress=show_progress
        )

        for epoch, _epoch_stat, info in trainer:
            self.logger.store(tab="train", cost_limit=self.cost_limit)
            if verbose:
                print(f"Epoch: {epoch}", info)

        return epoch, _epoch_stat, info


class OnpolicyAgent(BaseAgent):
    """在线策略智能体的基类。

    :meth:`~srl.agent.OnpolicyAgent.learn`: 函数经过定制以与\
        在线策略训练器配合工作。
    更多详细信息请参见 :class:`~fsrl.agent.BaseAgent`。
    """

    name = "OnpolicyAgent"

    def __init__(self) -> None:
        super().__init__()

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
        batch_size: int = 512,
        reward_threshold: float = 450,
        save_interval: int = 4,
        resume: bool = False,
        save_ckpt: bool = True,
        verbose: bool = True,
        show_progress: bool = True
    ) -> None:
        """在一组训练环境上训练策略。

        :param Union[gym.Env, BaseVectorEnv] train_envs: 用于训练策略的单个环境或
            向量化环境。
        :param Union[gym.Env, BaseVectorEnv] test_envs: 用于评估策略的单个环境或
            向量化环境，默认为None。
        :param int epoch: 训练epoch数，默认为300
        :param int episode_per_collect: 每次数据收集的回合数，
            默认为20
        :param int step_per_epoch: 每个训练epoch的步数，默认为
            10000
        :param int repeat_per_collect: 一个回合收集的策略更新重复次数，
            默认为4
        :param int buffer_size: 重放缓冲区的大小，默认为100000
        :param int testing_num: 测试期间评估的回合数，
            默认为2
        :param int batch_size: 训练的批次大小，对于
            :class:`~fsrl.agent.TRPOLagAgent` :class:`~fsrl.agent.CPOLagAgent` 默认为99999，
            对于其他为512
        :param float reward_threshold: 当平均奖励超过此阈值时停止训练的阈值，
            默认为450
        :param int save_interval: 保存策略的epoch数，默认为4
        :param bool resume: 是否从保存的检查点恢复训练，
            默认为False
        :param bool save_ckpt: 是否保存策略模型，默认为True
        :param bool verbose: 是否打印训练信息，默认为True
        :param bool show_progress: 是否显示tqdm训练进度条，
            默认为True
        """
        assert self.policy is not None, "策略未初始化"
        # 将策略设置为训练模式
        self.policy.train()
        # 收集器
        if isinstance(train_envs, gym.Env):
            buffer = ReplayBuffer(buffer_size)
        else:
            buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        train_collector = FastCollector(
            self.policy,
            train_envs,
            buffer,
            exploration_noise=True,
        )
        test_collector = FastCollector(
            self.policy, test_envs
        ) if test_envs is not None else None

        def stop_fn(reward, cost):
            return reward > reward_threshold and cost < self.cost_limit

        def checkpoint_fn():
            return {"model": self.state_dict}

        if save_ckpt:
            self.logger.setup_checkpoint_fn(checkpoint_fn)

        # 训练器
        trainer = OnpolicyTrainer(
            policy=self.policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epoch,
            batch_size=batch_size,
            cost_limit=self.cost_limit,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            episode_per_test=testing_num,
            episode_per_collect=episode_per_collect,
            stop_fn=stop_fn,
            logger=self.logger,
            resume_from_log=resume,
            save_model_interval=save_interval,
            verbose=verbose,
            show_progress=show_progress
        )

        for epoch, _epoch_stat, info in trainer:
            self.logger.store(tab="train", cost_limit=self.cost_limit)
            if verbose:
                print(f"Epoch: {epoch}", info)

        return epoch, _epoch_stat, info
