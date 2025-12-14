"""
此脚本用于评估 ev2gym 环境的性能。gdfgd
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V

from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2, eMPC_G2V_v2

from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

from ev2gym.baselines.heuristics import RoundRobin, RoundRobin_1transformer_powerlimit, ChargeAsFastAsPossible
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

from cost_functions import usrpenalty_cost, tr_overload_usrpenalty_cost, ProfitMax_TrPenalty_UserIncentives_safety

import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
import gymnasium as gym


def eval():
    """
    运行 ev2gym 环境的评估。
    """

    verbose = True
    
    # 开关：是否运行 Oracle 最优解计算
    # 注意：这需要完整的 Gurobi License，否则可能会报错 "Model too large"
    run_oracle = False

    # 如果有先前保存的 replay 文件（场景回放文件），可以在这里指定路径加载，以重现该场景。
    replay_path = "./replay/replay_sim_2024_07_05_106720.pkl"
    replay_path = None # 设置为 None 表示生成新的随机场景

    # 配置文件路径
    config_file = "ev2gym/example_config_files/V2GProfitMax.yaml"
    #config_file = "V2GProfit_base.yaml"

    # 初始化环境
    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=replay_path,
                 verbose=False,
                 #  seed=184692, # 固定随机种子以复现结果
                 reward_function=ProfitMax_TrPenalty_UserIncentives_safety, # 设置奖励函数
                 cost_function=tr_overload_usrpenalty_cost, # 设置成本/约束函数
                 save_replay=True, # 保存本次运行的场景，以便后续对比
                 save_plots=True, # 保存结果图表
                 )

    # 记录新生成的 replay 文件路径，用于后续的最优解计算
    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    state, _ = env.reset()

    # 获取生成的电动汽车（EV）配置信息
    ev_profiles = env.EVs_profiles
    # 计算所有EV中最长的停留时间
    max_time_of_stay = max([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])
    # 计算所有EV中最短的停留时间
    min_time_of_stay = min([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])

    print(f'电动汽车数量: {len(ev_profiles)}')
    print(f'最大停留时间: {max_time_of_stay}')
    print(f'最小停留时间: {min_time_of_stay}')

    # exit()
    # 选择要评估的 Agent（代理/策略）：
    # 下面注释掉的是各种不同的基准策略（MPC, Gurobi等）
    # agent = OCMF_V2G(env, control_horizon=30, verbose=True)
    # agent = OCMF_G2V(env, control_horizon=25, verbose=True)
    # agent = eMPC_V2G(env, control_horizon=15, verbose=False)
    # agent = PowerTrackingErrorrMin(new_replay_path)
    # agent = eMPC_G2V(env, control_horizon=15, verbose=False)
    # agent = eMPC_V2G_v2(env, control_horizon=10, verbose=False)
    # agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)
    
    # 当前使用的策略：尽快充满电 (ChargeAsFastAsPossible)
    # 这是一种简单的启发式策略，即插即充，满功率运行。
    agent = ChargeAsFastAsPossible(verbose=False)
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()
    
    rewards = []
    print(f'模拟日期: {env.sim_date}')
    
    # 开始主要的模拟循环
    for t in range(env.simulation_length):
        # 代理根据当前环境状态决定动作
        actions = agent.get_action(env)

        # 环境执行动作，返回新的状态、奖励、是否结束等信息
        new_state, reward, done, truncated, stats = env.step(
            actions, visualize=False)  # takes action
        rewards.append(reward)

        # print(stats['cost'])

        if done:
            print(stats)
            print(f'总利润: {stats["total_profits"]}')
            print(f'平均用户满意度: {stats["average_user_satisfaction"]}')
            print(f'模拟在步骤 {env.current_step} 结束')
            break

    # exit()
    # ---------------------------------------------------------
    # 使用 Oracle（上帝视角/最优解求解器）再跑一遍同样的场景
    # ---------------------------------------------------------
    
    # 开关：是否运行 Oracle 最优解计算
    # 注意：这需要完整的 Gurobi License，否则可能会报错 "Model too large"
    # run_oracle 配置已移至函数开头

    if run_oracle:
        # V2GProfitMaxOracleGB 使用 Gurobi 求解器，在已知未来的情况下计算理论最大利润
        agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)

        # 重新加载完全相同的环境场景 (通过 load_from_replay_path)
        env = EV2Gym(config_file=config_file,
                     load_from_replay_path=new_replay_path,
                     verbose=False,
                     save_plots=True,
                     )
        state, _ = env.reset()
        rewards_opt = []

        for t in range(env.simulation_length):
            actions = agent.get_action(env)
            # if verbose:
            #     print(f' OptimalActions: {actions}')

            new_state, reward, done, truncated, stats = env.step(
                actions, visualize=False)  # takes action
            rewards_opt.append(reward)

            if done:
                # print(stats)
                print(f'最优解 总利润: {stats["total_profits"]}')
                print(f'最优解 平均用户满意度: {stats["average_user_satisfaction"]}')
                print(f'奖励: {reward} \t 结束: {done}')

            if done:
                break



if __name__ == "__main__":
    counter = 0
    while True:
        print(f'============================= 计数器: {counter}')
        eval()
        counter += 1
        exit() # 跑完一次就退出
