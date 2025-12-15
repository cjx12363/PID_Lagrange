import numpy as np


def projection(x):
    """
    投影函数，确保值非负。
    相当于 ReLU 激活函数: max(0, x)
    """
    return np.maximum(0, x)


class LagrangianOptimizer(object):
    """
    基于 PID 控制器的拉格朗日乘子优化器，
    参考论文: https://proceedings.mlr.press/v119/stooke20a.html.

    :param List pid: PID 控制器的系数, 分别为 kp (比例), ki (积分), kd (微分).

    .. note::

        如果 kp 和 kd 为 0，则简化为标准的基于 SGD 的拉格朗日优化器。
    """

    def __init__(self, pid: tuple = (0.05, 0.0005, 0.1)) -> None:
        super().__init__()
        assert len(pid) == 3, "pid 参数应该是一个包含 3 个数字的列表或元组"
        self.pid = tuple(pid)
        self.error_old = 0.
        self.error_integral = 0.
        self.lagrangian = 0.

    def step(self, value: float, threshold: float) -> None:
        """优化乘子一步

        :param float value: 当前的估算值 (例如当前的 cost)
        :param float threshold: 该值的阈值 (例如 cost limit)
        """
        # 计算误差: 当前值 - 阈值。
        # 如果当前值(cost) > 阈值(limit)，error_new > 0，拉格朗日乘子应当增大以通过惩罚项抑制 cost
        error_new = np.mean(value - threshold)  # [batch]
        
        # 误差的微分项 (D): 本次误差 - 上次误差，反映误差变化趋势
        error_diff = projection(error_new - self.error_old)
        
        # 误差的积分项 (I): 误差累积和，用于消除稳态误差
        self.error_integral = projection(self.error_integral + error_new)
        
        self.error_old = error_new
        
        # PID 更新公式: Kp * P + Ki * I + Kd * D
        # 使用 projection (ReLU) 保证拉格朗日乘子非负，符合对偶问题性质
        self.lagrangian = projection(
            self.pid[0] * error_new + self.pid[1] * self.error_integral +
            self.pid[2] * error_diff
        )

    def get_lag(self) -> float:
        """获取拉格朗日乘子。"""
        return self.lagrangian

    def state_dict(self) -> dict:
        """获取该拉格朗日优化器的参数"""
        params = {
            "pid": self.pid,
            "error_old": self.error_old,
            "error_integral": self.error_integral,
            "lagrangian": self.lagrangian
        }
        return params

    def load_state_dict(self, params: dict) -> None:
        """加载参数以继续训练"""
        self.pid = params["pid"]
        self.error_old = params["error_old"]
        self.error_integral = params["error_integral"]
        self.lagrangian = params["lagrangian"]
