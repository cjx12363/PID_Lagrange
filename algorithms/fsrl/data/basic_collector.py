import time
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_numpy

from fsrl.data.traj_buf import TrajectoryBuffer
from fsrl.policy import BasePolicy


class BasicCollector:
    """单环境基础收集器。

    此收集器不支持向量环境，仅用于实验目的。支持使用网格过滤器将数据存储在轨迹缓冲区中，
    可用于高效收集轨迹级交互数据集。
    """

    def __init__(
        self,
        policy: BasePolicy,  # 策略实例
        env: gym.Env,  # Gym环境
        buffer: Optional[ReplayBuffer] = None,  # 重放缓冲区（None表示不存储数据）
        exploration_noise: Optional[bool] = False,  # 是否添加探索噪声
        traj_buffer: Optional[TrajectoryBuffer] = None,  # 轨迹缓冲区
    ):
        self.env = env
        self.policy = policy
        if buffer is None:
            buffer = ReplayBuffer(1)

        self.buffer = buffer
        self.exploration_noise = exploration_noise
        self._action_space = self.env.action_space

        self.traj_buffer = traj_buffer
        self.reset(False)

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
            info={}
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """重置数据缓冲区。"""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_stat(self) -> None:
        """重置统计变量。"""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """重置所有环境。"""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(**gym_reset_kwargs)
        returns_info = isinstance(rval,
                                  (tuple, list
                                   )) and len(rval) == 2 and isinstance(rval[1], dict)
        if returns_info:
            obs, info = rval
            self.data.info = [info]
        else:
            obs = rval
        self.data.obs = [obs]

    def collect(
        self,
        n_episode: int = 0,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """收集指定数量的episode。"""
        start_time = time.time()

        step_count = 0
        total_cost = 0
        termination_count = 0
        truncation_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []

        while True:
            # 获取下一个动作
            if random:
                act_sample = self._action_space.sample()
                act_sample = self.policy.map_action_inverse(act_sample)
                self.data.update(act=[act_sample])
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        result = self.policy(
                            Batch(obs=self.data.obs, info=self.data.info)
                        )

                else:
                    result = self.policy(Batch(obs=self.data.obs, info=self.data.info))

                act = to_numpy(result.act)[0]
                # print(act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(act=[act])

            # 首先获取有界和重映射的动作（不保存到缓冲区）
            action_remap = np.squeeze(self.policy.map_action(self.data.act))
            # 在环境中执行步骤
            result = self.env.step(action_remap)
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

            cost = info.get("cost", 0)

            self.data.update(
                obs_next=[obs_next],
                rew=[rew],
                terminated=[terminated],
                truncated=[truncated],
                done=[done],
                cost=[cost],
                info=[info]
            )

            termination_count += terminated
            truncation_count += truncated

            total_cost += cost

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # 将数据添加到缓冲区
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data, 1)

            if self.traj_buffer is not None:
                traj_data = Batch(
                    observations=self.data.obs,
                    next_observations=self.data.obs_next,
                    actions=[action_remap],
                    rewards=self.data.rew,
                    costs=self.data.cost,
                    terminals=self.data.terminated,
                    timeouts=self.data.truncated
                )
                self.traj_buffer.store(traj_data)

            step_count += 1

            if done:
                episode_count += 1
                episode_lens.append(ep_len)
                episode_rews.append(ep_rew)
                # 现在将obs_next复制到obs，但由于可能有已完成的episode，
                # 我们必须先重置已完成的环境
                self.reset_env(gym_reset_kwargs)

            self.data.obs = self.data.obs_next

            if episode_count >= n_episode:
                break

        # 生成统计信息
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        self.reset_env()

        done_count = truncation_count + termination_count

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rew": np.mean(episode_rews),
            "len": np.mean(episode_lens),
            "total_cost": total_cost,
            "cost": total_cost / episode_count,
            "truncated": truncation_count / done_count,
            "terminated": termination_count / done_count,
        }
