#from GF.action_wrapper import ThreeStep_Action_DiscreteActionSpace, mask_fn, Fully_Discrete

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from environments.models.ev2gym_env import EV2Gym
from environments.rl_integration.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from environments.rl_integration.reward import profit_maximization

from environments.rl_integration.cost_functions import tr_overload_usrpenalty_cost, usrpenalty_cost, ProfitMax_TrPenalty_UserIncentives_safety, paper_cost_function

from environments.rl_integration.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml
import random
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from environments.rl_integration.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from environments.rl_integration.reward import profit_maximization, paper_reward_function
from environments.rl_integration.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import pprint
from dataclasses import asdict

from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb


from algorithms.fsrl.data import FastCollector
from algorithms.fsrl.agent import SACLagAgent, PPOLagAgent, CPOAgent, CVPOAgent
from algorithms.fsrl.policy import CVPO
from algorithms.fsrl.trainer import OffpolicyTrainer
from algorithms.fsrl.utils import TensorboardLogger, WandbLogger
from algorithms.fsrl.utils.exp_util import auto_name, seed_all
from algorithms.fsrl.utils.net.common import ActorCritic
from algorithms.fsrl.utils.net.continuous import DoubleCritic, SingleCritic

from dataclasses import dataclass
from typing import Tuple

from gymnasium import Wrapper

class SpecMaxStepsWrapper(Wrapper):
    def __init__(self, env, spec_max_steps):
        """
        初始化包装器

        参数:
        - env: 要包装的环境
        - spec_max_steps: 环境允许的最大步数
        """
        super(SpecMaxStepsWrapper, self).__init__(env)
        self.spec.max_episode_steps = spec_max_steps

# ==================== 训练配置 (手动修改区) ====================
TRAIN_ALGORITHM = "sacl"      # 训练算法: 'cpo', 'cvpo', 'ppol', 'sacl'
COST_LIMIT = 2                # 成本限制
EPOCH = 400                  # 训练轮数
TRAIN_NUM = 10                # 训练环境数量 (建议不超过CPU核心数)
TEST_NUM = 10                 # 测试环境数量 (建议不超过CPU核心数)
SEED = 10                     # 随机种子
# ===============================================================

# 环境配置
config_file = "environments/config/V2GProfit_base.yaml"
# config_file = "V2GProfit_loads.yaml"

if config_file == "environments/config/V2GProfit_base.yaml":
        state_function = V2G_profit_max
        cost_function = paper_cost_function

if config_file == "V2GProfit_loads.yaml":
        state_function = V2G_profit_max_loads
        cost_function = paper_cost_function

reward_function = paper_reward_function
cost_function = paper_cost_function

# 注册自定义环境
gym.envs.register(
id='fsrl-v0',
entry_point='environments.models.ev2gym_env:EV2Gym',
kwargs={
        'config_file': config_file,
        'verbose': False,
        'save_plots': False,
        'generate_rnd_game': True,
        'reward_function': reward_function,
        'state_function': state_function,
        'cost_function': cost_function,
},
)

task = "fsrl-v0"

def make_env(task_id, spec_max_steps=None):
        """工厂函数，用于创建环境实例，避免lambda闭包问题"""
        def _init():
                env = gym.make(task_id)
                if spec_max_steps is not None:
                        return SpecMaxStepsWrapper(env, spec_max_steps)
                return env
        return _init

def train_cpo(args):

        seed: int = args.seed
        cost_limit: int = args.cost_limit
        epoch: int = args.epoch
        training_num: int = args.train_num
        testing_num: int = args.test_num
        step_per_epoch: int = 3000
        gamma: float = 0.99
        thread: int = 1
        buffer_size: int = 200000

        run_name= f'cpo_exp1_1_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name =  f'CPO_5spawn_20cs_cost_lim_{cost_limit}_epochs_{epoch}_usr_1000_train_envs_8_test_envs_8_run{random.randint(0, 1000)}'
        group_name = 'EXP1_5'                   

        wandb.init(project='experiment_1_5',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        )

        task = "fsrl-v0"

        # 初始化日志记录器
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_5/cpo", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)

        # CPO 智能体
        agent = CPOAgent(env=env, logger=logger, cost_limit=cost_limit, seed=seed, thread=thread, gamma=gamma)

        train_envs = SubprocVectorEnv([make_env(task) for _ in range(training_num)])
        test_envs = SubprocVectorEnv([make_env(task) for _ in range(testing_num)])

        agent.learn(train_envs=train_envs, buffer_size = buffer_size, test_envs=test_envs, epoch=epoch, testing_num=testing_num,
                    episode_per_collect=training_num, step_per_epoch=step_per_epoch, save_interval=1)

def train_cvpo(args):

        # 通用任务参数
        task: str = "fsrl-v0"
        cost_limit: int = args.cost_limit
        device: str = "cpu"
        thread: int = 1  # 使用CPU训练时设为1
        seed: int = args.seed
        # CVPO 算法参数
        estep_iter_num: int = 1
        estep_kl: float = 0.02
        estep_dual_max: float = 20
        estep_dual_lr: float = 0.02
        sample_act_num: int = 16
        mstep_iter_num: int = 1
        mstep_kl_mu: float = 0.005
        mstep_kl_std: float = 0.0005
        mstep_dual_max: float = 0.5
        mstep_dual_lr: float = 0.1
        actor_lr: float = 5e-4
        critic_lr: float = 1e-3
        gamma: float = 0.99 # was 0.97
        n_step: int = 2
        tau: float = 0.05
        hidden_sizes: Tuple[int, ...] = (128, 128)
        double_critic: bool = False
        conditioned_sigma: bool = True
        unbounded: bool = False
        last_layer_scale: bool = False
        # 数据采集参数
        epoch: int = args.epoch
        episode_per_collect: int = args.train_num
        step_per_epoch: int = 3000
        update_per_step: float = 0.2
        buffer_size: int = 200000
        worker: str = "ShmemVectorEnv"
        training_num: int = args.train_num
        testing_num: int = args.test_num
        # 通用训练参数
        batch_size: int = 256
        reward_threshold: float = 10000  # 用于提前停止
        save_interval: int = 1
        deterministic_eval: bool = True
        action_scaling: bool = True
        action_bound_method: str = "clip"
        resume: bool = False
        save_ckpt: bool = True  # 设为True保存策略模型
        verbose: bool = False
        render: bool = False

        # 日志参数
        group_name: str = "EXP1_5"
        # run_name= f'cvpo_v45_step_epoch_9k_buff_200k_5spawn_10cs_90kw_loads_0_6_PV_0_1_seed{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name= f'cvpo_v67_6_h28_20_usr_5spawn_10cs_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name= f'cvpo_step_epoch_3k_30chargers_buff_400k_exp1_3_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        run_name= f'cvpo_const_eff_0_9_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'

        wandb.init(project='experiment_1_5',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        )

        # 初始化日志记录器
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_5", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)

        sim_length = env.env.env.simulation_length

        agent = CVPOAgent(
                env=SpecMaxStepsWrapper(gym.make(task), sim_length),
                logger=logger,
                cost_limit=cost_limit,
                device=device,
                thread=thread,
                seed=seed,
                estep_iter_num=estep_iter_num,
                estep_kl=estep_kl,
                estep_dual_max=estep_dual_max,
                estep_dual_lr=estep_dual_lr,
                sample_act_num=sample_act_num,
                mstep_iter_num=mstep_iter_num,
                mstep_kl_mu=mstep_kl_mu,
                mstep_kl_std=mstep_kl_std,
                mstep_dual_max=mstep_dual_max,
                mstep_dual_lr=mstep_dual_lr,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                n_step=n_step,
                tau=tau,
                hidden_sizes=hidden_sizes,
                double_critic=double_critic,
                conditioned_sigma=conditioned_sigma,
                unbounded=unbounded,
                last_layer_scale=last_layer_scale,
                deterministic_eval=deterministic_eval,
                action_scaling=action_scaling,
                action_bound_method=action_bound_method,
                lr_scheduler=None
        )

        # training_num = min(training_num, episode_per_collect)
        train_envs = SubprocVectorEnv([make_env(task, sim_length) for _ in range(training_num)])
        test_envs = SubprocVectorEnv([make_env(task, sim_length) for _ in range(testing_num)])

        # 开始训练
        agent.learn(
                train_envs=train_envs,
                test_envs=test_envs,
                epoch=epoch,
                episode_per_collect=episode_per_collect,
                step_per_epoch=step_per_epoch,
                update_per_step=update_per_step,
                buffer_size=buffer_size,
                testing_num=testing_num,
                batch_size=batch_size,
                reward_threshold=reward_threshold,  # 用于提前停止
                save_interval=save_interval,
                resume=resume,
                save_ckpt=save_ckpt,  # 设为True保存策略模型
                verbose=verbose,
        )

def train_ppol(args):
        # 通用任务参数
        task: str = "fsrl-v0"
        cost_limit: int = args.cost_limit
        device: str = "cpu"
        thread: int = 1  # 使用CPU训练时设为1
        seed: int = args.seed
        # 算法参数
        lr: float = 5e-4
        hidden_sizes: Tuple[int, ...] = (128, 128)
        unbounded: bool = False
        last_layer_scale: bool = False
        # PPO 特定参数
        target_kl: float = 0.08
        vf_coef: float = 0.25
        max_grad_norm: float = 0.5
        gae_lambda: float = 0.95
        eps_clip: float = 0.2
        dual_clip: float = None
        value_clip: bool = False  # 不需要
        norm_adv: bool = True  # 有助于提高训练稳定性
        recompute_adv: bool = False
        # 拉格朗日特定参数
        use_lagrangian: bool = True
        #lagrangian_pid: Tuple[float, ...] = (0.05, 0.005, 0.1)
        lagrangian_pid: Tuple[float, ...] = (0, 0.002, 0)
        rescaling: bool = True
        # 基础策略通用参数
        gamma: float = 0.99
        max_batchsize: int = 100000
        rew_norm: bool = False  # 不需要，会减慢训练并降低最终性能
        deterministic_eval: bool = True
        action_scaling: bool = True
        action_bound_method: str = "clip"
        # 数据采集参数
        epoch: int = args.epoch
        episode_per_collect: int = args.train_num
        step_per_epoch: int = 3000
        repeat_per_collect: int = 8  # 增大可提高效率，但稳定性降低
        buffer_size: int = 200000
        worker: str = "ShmemVectorEnv"
        training_num: int = args.train_num
        testing_num: int = args.test_num
        # 通用参数
        batch_size: int = 256
        reward_threshold: float = 10000  # 用于提前停止
        save_interval: int = 1
        resume: bool = False
        save_ckpt: bool = True  # 设为True保存策略模型
        verbose: bool = False
        render: bool = False

        # 日志参数
        group_name: str = "EXP1_5"
        # run_name= f'PPOL_h20_1powerlimit_sacl_100_v2g_cost_40_loads_PV_no_DR_5spawn_10cs_90kw_seed_{seed}_cost_lim_{int(cost_limit)}_usr_-3_100_tr_30_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        run_name= f'ppol_10repeatpercollect_exp1_1_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'

        wandb.init(project='PPO-L',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        )

        # 初始化日志记录器
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_5/ppol", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)

        agent = PPOLagAgent(
        env=env,
        logger=logger,
        device=device,
        thread=thread,
        seed=seed,
        lr=lr,
        hidden_sizes=hidden_sizes,
        unbounded=unbounded,
        last_layer_scale=last_layer_scale,
        target_kl=target_kl,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        dual_clip=dual_clip,
        value_clip=value_clip,
        advantage_normalization=norm_adv,
        recompute_advantage=recompute_adv,
        use_lagrangian=use_lagrangian,
        lagrangian_pid=lagrangian_pid,
        cost_limit=cost_limit,
        rescaling=rescaling,
        gamma=gamma,
        max_batchsize=max_batchsize,
        reward_normalization=rew_norm,
        deterministic_eval=deterministic_eval,
        action_scaling=action_scaling,
        action_bound_method=action_bound_method,
    )

        train_envs = SubprocVectorEnv([make_env(task) for _ in range(training_num)])
        test_envs = SubprocVectorEnv([make_env(task) for _ in range(testing_num)])

        # 开始训练
        agent.learn(
                train_envs=train_envs,
                test_envs=test_envs,
                epoch=epoch,
                episode_per_collect=episode_per_collect,
                step_per_epoch=step_per_epoch,
                repeat_per_collect=repeat_per_collect,
                buffer_size=buffer_size,
                testing_num=testing_num,
                batch_size=batch_size,
                reward_threshold=reward_threshold,
                save_interval=save_interval,
                resume=resume,
                save_ckpt=save_ckpt,
                verbose=verbose,
        )


def train_sacl(args):
        # 通用任务参数
        task: str = "fsrl-v0"
        cost_limit: int = args.cost_limit
        device: str = "cpu"
        thread: int = 1  # 使用CPU训练时设为1
        seed: int = args.seed
        # 算法参数
        actor_lr: float = 5e-4
        critic_lr: float = 1e-3
        hidden_sizes: Tuple[int, ...] = (128, 128)
        auto_alpha: bool = True
        alpha_lr: float = 3e-4
        alpha: float = 0.005
        tau: float = 0.05
        n_step: int = 2
        conditioned_sigma: bool = True
        unbounded: bool = False
        last_layer_scale: bool = False
        # 拉格朗日特定参数
        use_lagrangian: bool = True
        lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
        rescaling: bool = True
        # 基础策略通用参数
        gamma: float = 0.99
        deterministic_eval: bool = True
        action_scaling: bool = True
        action_bound_method: str = "clip"
        # 数据采集参数
        epoch: int = args.epoch
        episode_per_collect: int = args.train_num
        step_per_epoch: int = 3000
        update_per_step: float = 0.2
        buffer_size: int = 400000
        worker: str = "ShmemVectorEnv"
        training_num: int = args.train_num
        testing_num: int = args.test_num
        # 通用训练参数
        batch_size: int = 256
        reward_threshold: float = 10000  # 用于提前停止
        save_interval: int = 1
        resume: bool = False
        save_ckpt: bool = True  # 设为True保存策略模型
        verbose: bool = True
        render: bool = False

        # 日志参数
        group_name: str = "EXP1_3"
        # run_name= f'sacl_v4_h20_no_v2g_cost_5spawn_10cs_90kw_cost_lim_{int(cost_limit)}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        # run_name= f'sacl_exp1_1_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        run_name= f'sacl_step_epoch_3k_30chargers_buff_400k_exp1_3_seed_{seed}_cost_lim_{cost_limit}_train_envs_{training_num}_test_envs_{testing_num}_run{random.randint(0, 1000)}'
        

        wandb.init(project='experiment_1_3',
                        sync_tensorboard=True,
                        group=group_name,
                        name=run_name,
                        save_code=True,
                        )

        # 初始化日志记录器
        logger = WandbLogger(log_dir="fsrl_logs/EXP1_3", log_txt=True, group=group_name, name=run_name)

        env = gym.make(task)

        sim_length = env.env.env.simulation_length

        agent = SACLagAgent(
        env=SpecMaxStepsWrapper(gym.make(task), sim_length),
        logger=logger,
        # 通用任务参数
        device=device,
        thread=thread,
        seed=seed,
        # 算法参数
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        hidden_sizes=hidden_sizes,
        auto_alpha=auto_alpha,
        alpha_lr=alpha_lr,
        alpha=alpha,
        tau=tau,
        n_step=n_step,
        # 拉格朗日特定参数
        use_lagrangian=use_lagrangian,
        lagrangian_pid=lagrangian_pid,
        cost_limit=cost_limit,
        rescaling=rescaling,
        # 基础策略通用参数
        gamma=gamma,
        conditioned_sigma=conditioned_sigma,
        unbounded=unbounded,
        last_layer_scale=last_layer_scale,
        deterministic_eval=deterministic_eval,
        action_scaling=action_scaling,
        action_bound_method=action_bound_method,
        lr_scheduler=None
    )


        # training_num = min(training_num, episode_per_collect)
        train_envs = SubprocVectorEnv([make_env(task, sim_length) for _ in range(training_num)])
        test_envs = SubprocVectorEnv([make_env(task, sim_length) for _ in range(testing_num)])
        

        agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=epoch,
        episode_per_collect=episode_per_collect,
        step_per_epoch=step_per_epoch,
        update_per_step=update_per_step,
        buffer_size=buffer_size,
        testing_num=testing_num,
        batch_size=batch_size,
        reward_threshold=reward_threshold,  # for early stop purpose
        save_interval=save_interval,
        resume=resume,
        save_ckpt=save_ckpt,
        verbose=verbose,
    )


if __name__ == "__main__":
        # 使用文件顶部的硬编码配置，也支持命令行覆盖
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", type=str, default=TRAIN_ALGORITHM)
        parser.add_argument("--cost_limit", type=int, default=COST_LIMIT)
        parser.add_argument("--epoch", type=int, default=EPOCH)
        parser.add_argument("--train_num", type=int, default=TRAIN_NUM)
        parser.add_argument("--test_num", type=int, default=TEST_NUM)
        parser.add_argument("--config", type=str, default=config_file)
        parser.add_argument("--seed", type=int, default=SEED)
        args = parser.parse_args()

        if args.train == "cvpo":
                train_cvpo(args)
        elif args.train == "cpo":
                train_cpo(args)
        elif args.train == "ppol":
                train_ppol(args)
        elif args.train == "sacl":
                train_sacl(args)
        else:
                print("Invalid training algorithm. Please choose either 'cpo', 'cvpo', 'ppol' or 'sacl'")