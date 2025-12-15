import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import (
    Batch,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.env import BaseVectorEnv, DummyVectorEnv

from fsrl.policy import BasePolicy


class FastCollector(object):
    """收集器使策略能够与不同类型的环境进行准确数量的episode交互。

    这是Tianshou收集器的简化版本，主要变化是支持从交互数据中提取成本信号。
    """

    def __init__(
        self,
        policy: BasePolicy,  # 策略实例
        env: Union[gym.Env, BaseVectorEnv],  # 环境
        buffer: Optional[ReplayBuffer] = None,  # 重放缓冲区
        preprocess_fn: Optional[Callable[..., Batch]] = None,  # 预处理函数
        exploration_noise: bool = False,  # 是否添加探索噪声
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            self.env = DummyVectorEnv([lambda: env])
        else:
            self.env = env
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = self.env.action_space
        # 避免在__init__外部创建属性
        self.reset(False)

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """检查缓冲区是否符合约束。"""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if isinstance(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """重置环境、统计信息、当前数据和重放内存。"""
        # 使用空Batch作为"state"，使self.data支持切片
        # 将空Batch转换为None传递给策略
        self.data = Batch(
            obs={},
            act={},
            rew={},
            cost={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={}
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """重置统计变量。"""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """重置数据缓冲区。"""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """重置所有环境。"""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(**gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)
        )
        if returns_info:
            obs, info = rval
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs, info=info, env_id=np.arange(self.env_num)
                )
                obs = processed_data.get("obs", obs)
                info = processed_data.get("info", info)
            self.data.info = info
        else:
            obs = rval
            if self.preprocess_fn:
                obs = self.preprocess_fn(obs=obs,
                                         env_id=np.arange(self.env_num)).get("obs", obs)
        self.data.obs = obs

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """重置隐藏状态。"""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(global_ids, **gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)
        )
        if returns_info:
            obs_reset, info = rval
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs_reset, info=info, env_id=global_ids
                )
                obs_reset = processed_data.get("obs", obs_reset)
                info = processed_data.get("info", info)
            self.data.info[local_ids] = info
        else:
            obs_reset = rval
            if self.preprocess_fn:
                obs_reset = self.preprocess_fn(obs=obs_reset,
                                               env_id=global_ids).get("obs", obs_reset)
        self.data.obs_next[local_ids] = obs_reset

    def collect(
        self,
        n_episode: int = 1,
        random: bool = False,
        render: bool = False,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """收集指定数量的episode。"""
        if n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError("Please specify n_episode"
                            "in FastCollector.collect().")

        start_time = time.time()

        step_count = 0
        total_cost = 0
        termination_count = 0
        truncation_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # 恢复状态：如果最后一个状态是None，它不会存储
            last_state = self.data.policy.pop("hidden_state", None)

            # 获取下一个动作
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool的动作空间不是每环境的
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # 比retain_grad版本更快
                        # self.data.obs将被智能体用来获取结果
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # 将state/act/policy更新到self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # 将状态保存到缓冲区
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # 首先获取有界和重映射的动作（不保存到缓冲区）
            action_remap = self.policy.map_action(self.data.act)
            # 在环境中执行步骤
            result = self.env.step(action_remap, ready_env_ids)
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            elif len(result) == 4:
                obs_next, rew, done, info = result
                if isinstance(info, dict):
                    truncated = info["TimeLimit.truncated"]
                else:
                    truncated = np.array(
                        [
                            info_item.get("TimeLimit.truncated", False)
                            for info_item in info
                        ]
                    )
                terminated = np.logical_and(done, ~truncated)
            else:
                raise ValueError()

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            cost = self.data.info.get("cost", np.zeros(rew.shape))
            total_cost += np.sum(cost)
            self.data.update(cost=cost)

            if render:
                self.env.render()

            # 将数据添加到缓冲区
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # 收集统计信息
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                termination_count += np.sum(terminated)
                truncation_count += np.sum(truncated)
                # 现在将obs_next复制到obs，但由于可能有已完成的episode，
                # 我们必须先重置已完成的环境
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

                # 从 ready_env_ids中删除多余的环境id以避免选择环境时的偏差
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if n_episode and episode_count >= n_episode:
                break

        # 生成统计信息
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                cost={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens = list(map(np.concatenate, [episode_rews, episode_lens]))
            rew_mean = rews.mean()
            len_mean = lens.mean()
        else:
            rew_mean = len_mean = 0

        done_count = termination_count + truncation_count

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rew": rew_mean,
            "len": len_mean,
            "total_cost": total_cost,
            "cost": total_cost / episode_count,
            "truncated": truncation_count / done_count,
            "terminated": termination_count / done_count,
        }
