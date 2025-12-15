import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
from tianshou.data import Batch
from tianshou.data.utils.converter import to_hdf5


class TrajectoryBuffer:
    """存储训练期间收集的轨迹的缓冲区。

    如果使用网格过滤器，它会根据成本-回报和奖励-回报空间上的密度丢弃超出的轨迹。
    """

    def __init__(
        self,
        max_trajectory: int = 99999,  # 最大轨迹数
        use_grid_filter: bool = True,  # 是否使用网格过滤器
        rmin: float = -np.inf,  # 最小奖励回报
        rmax: float = np.inf,  # 最大奖励回报
        cmin: float = -np.inf,  # 最小成本回报
        cmax: float = np.inf,  # 最大成本回报
        filter_interval: float = 2  # 过滤间隔比例
    ):
        self.max_trajectory = max_trajectory
        self.buffer: List[Batch] = []
        self.current_trajectory = Batch()
        self.current_rew, self.current_cost = 0, 0
        self.metrics: List[np.ndarray] = []
        self.rmin = rmin
        self.rmax = rmax
        self.cmin = cmin
        self.cmax = cmax

        self.use_grid_filter = use_grid_filter
        if self.use_grid_filter:
            assert filter_interval > 1, "the filter interval should be greater than 1"
            self.filtering_thres = int(filter_interval * max_trajectory)

    def store(self, data: Batch) -> None:
        """在缓冲区中存储一批数据。"""
        # 将数据连接到当前轨迹
        self.current_trajectory = Batch.cat([self.current_trajectory, data])
        done = data["terminals"].item() or data["timeouts"].item()
        self.current_rew += data["rewards"].item()
        self.current_cost += data["costs"].item()
        if done:
            if self.current_rew > self.rmax or self.current_rew < self.rmin \
                    or self.current_cost > self.cmax or self.current_cost < self.cmin:
                pass
            else:
                if len(self.buffer) < self.max_trajectory:
                    self.buffer.append(self.current_trajectory)
                    self.metrics.append(np.array([self.current_rew, self.current_cost]))
                else:
                    if self.use_grid_filter:
                        self.buffer.append(self.current_trajectory)
                        self.metrics.append(
                            np.array([self.current_rew, self.current_cost])
                        )
                        # 当缓冲区大小达到filtering_thres时应用网格过滤器
                        if len(self.buffer) >= self.filtering_thres:
                            self.apply_grid_filter()
                    else:
                        idx_to_replace = np.random.randint(0, len(self.buffer))
                        self.buffer[idx_to_replace] = self.current_trajectory
                        self.metrics[idx_to_replace] = np.array(
                            [self.current_rew, self.current_cost]
                        )
            self.current_trajectory = Batch()
            self.current_rew, self.current_cost = 0, 0

    def apply_grid_filter(self) -> None:
        """对缓冲区和指标数据应用网格过滤。

        过滤器将删除一些密度最高的轨迹。
        """
        kept_idxs = self.filter_points(self.metrics, self.max_trajectory)
        # 保留kept_idxs中的数据并删除其他数据；原地操作
        indices_set = set(kept_idxs)
        write_index = 0

        for read_index in range(len(self.buffer)):
            if read_index in indices_set:
                if read_index != write_index:
                    self.buffer[write_index] = self.buffer[read_index]
                    self.metrics[write_index] = self.metrics[read_index]
                write_index += 1

        del self.buffer[write_index:]
        del self.metrics[write_index:]

    @staticmethod
    def filter_points(points: list, target_size: int) -> list:
        """过滤 2D点列表并返回过滤后的索引列表。"""
        points = np.array(points)
        grid_size = int(np.ceil(np.sqrt(target_size)))
        # 创建网格来存储频率
        grid_range = [(points[:, i].min(), points[:, i].max()) for i in range(2)]
        cell_size = [(r[1] - r[0]) / grid_size for r in grid_range]

        grid = defaultdict(list)
        for i, point in enumerate(points):
            cell = tuple(
                int((point[i] - grid_range[i][0]) // cell_size[i]) for i in range(2)
            )
            grid[cell].append(i)

        kept_idxs = []
        # 首先，从每个非空单元格中添加一个点
        for pt_idxs in grid.values():
            if len(pt_idxs) > 0:
                idx = pt_idxs.pop()
                kept_idxs.append(idx)

        # 如果减少后的点数少于target_size，则添加更多点
        non_empty_cells = [cell for cell, points in grid.items() if len(points) > 0]
        while len(kept_idxs) < target_size:
            cell = random.choice(non_empty_cells)
            idx = grid[cell].pop()
            kept_idxs.append(idx)
            if len(grid[cell]) == 0:
                non_empty_cells.remove(cell)

        return kept_idxs[:target_size]

    def __len__(self) -> int:
        return sum([len(traj) for traj in self.buffer])

    def sample(self, batch_size: int) -> Batch:
        """从缓冲区中采样一批转换。"""
        num_trajectories = len(self.buffer)
        traj_indices = np.random.randint(0, num_trajectories, size=batch_size)
        sampled_batch = Batch()
        for i in range(batch_size):
            sampled_traj = self.buffer[traj_indices[i]]
            transition_idx = np.random.randint(0, len(sampled_traj))
            sampled_transition = sampled_traj[transition_idx]
            Batch.cat(sampled_batch, sampled_transition)
        return sampled_batch

    def get_all(self) -> Batch:
        """将缓冲区中存储的所有转换作为单个batch返回。"""
        return Batch.cat(self.buffer)

    def save(self, log_dir: str, dataset_name: str = "dataset.hdf5") -> None:
        """将整个缓冲区保存到磁盘为HDF5文件。"""
        print("Saving dataset...")
        if not os.path.exists(log_dir):
            print(f"Creating saving dir {log_dir}")
            os.makedirs(log_dir)
        dataset_path = os.path.join(log_dir, dataset_name)
        all_data = self.get_all()
        with h5py.File(dataset_path, "w") as f:
            to_hdf5(all_data, f, compression='gzip')
        print(f"Finish saving dataset to {dataset_path}!")
