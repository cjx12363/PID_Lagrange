'''本文件包含 RL 代理的各种示例奖励函数。用户可以在此处或使用如下相同结构在自己的文件中创建自己的奖励函数
'''

import math

def SquaredTrackingErrorReward(env,*args):
    '''此奖励函数使用功率设定值和充电功率潜力的最小值作为平方跟踪误差
    奖励为负值'''
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
        env.current_power_usage[env.current_step-1])**2
        
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    ''' 此奖励函数使用功率设定值和充电功率潜力的最小值作为平方跟踪误差
    它惩罚过载的变压器
    奖励为负值'''
    
    tr_max_limit = env.transformers[0].max_power[env.current_step-1]
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
        env.current_power_usage[env.current_step-1])**2
            
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
        
    for score in user_satisfaction_list:
        reward -= 1000 * (1 - score)
                    
    return reward

def ProfitMax_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs                      
    
    for score in user_satisfaction_list:        
        reward -= 1000*math.exp(-3*score) + 1000*math.exp(-3)
        
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        reward -= 1000*math.exp(-3*score) + 1000*math.exp(-3)
        
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    ''' 此奖励函数使用功率设定值和充电功率潜力的最小值作为平方跟踪误差
    奖励为负值
    如果 EV 未充电，则会惩罚奖励
    '''
    if env.current_power_usage[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2 - 100
    else:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    '''此奖励函数不考虑充电功率潜力'''
    
    reward = - (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    ''' 此奖励函数最小化跟踪器盈余并给予充电奖励 '''
    
    reward = 0
    if env.power_setpoints[env.current_step-1] < env.current_power_usage[env.current_step-1]:
            reward -= (env.current_power_usage[env.current_step-1]-env.power_setpoints[env.current_step-1])**2

    reward += env.current_power_usage[env.current_step-1] #/75
    
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    ''' 此奖励函数用于利润最大化的情况 '''
    
    reward = total_costs
    
    for score in user_satisfaction_list:
        # reward -= 100 * (1 - score)
        reward -= 100 * math.exp(-10*score)
    
    return reward

# ==========================================
# 黄金配置函数 1: Reward (目标函数)
# 目标：最大化利润，同时尽量让用户满意（软约束）
# ==========================================
def paper_reward_function(env, total_costs, user_satisfaction_list, *args):
    """
    Reward = 财务利润 (V2G) - 用户不满意惩罚
    注意：这里绝对不要包含变压器过载惩罚！安全问题交给 Cost 处理。
    """
    # 1. 基础奖励：财务利润 (Total Costs 在代码中通常代表净收益)
    # 如果 total_costs 是正数代表赚钱，负数代表花钱
    reward = total_costs 

    # 2. 用户满意度 (软约束)
    # 如果不加这个，Agent 会为了赚钱把电池放空，导致用户体验为 0
    # 我们希望 Agent 在赚钱的同时，尽量把车充满
    user_penalty = 0
    a = 2   # 惩罚系数 (可以调整，比如 10-50)
    b = -3   # 衰减系数
    for score in user_satisfaction_list:  
        # score 是 0-1 之间的数，1 代表非常满意
        # 当 score=1 时，惩罚接近 0；当 score=0 时，惩罚最大
        # 修改：减去 a*math.exp(b) 以确保当 score=1 时惩罚为 0
        user_penalty += a * math.exp(b * score) - a * math.exp(b)
    
    # 总奖励 = 钱 - 被骂
    return reward - user_penalty