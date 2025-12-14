'''
本文件包含 EV2Gym 安全环境的成本和奖励函数。
'''
import math

def usrpenalty_cost(env, total_costs, user_satisfaction_list, *args):
    """
    返回用户不满意惩罚的成本。
    """

    cost = 0
    
    a = 20
    b = -3
    
    for score in user_satisfaction_list:  
        cost += a*math.exp(b*score) - a*math.exp(b)

    return cost


def tr_overload_usrpenalty_cost(env, total_costs, user_satisfaction_list, *args):
    """
    返回变压器过载和用户不满意惩罚的成本。
    """

    cost = 0

    for tr in env.transformers:
        cost += 50 * tr.get_how_overloaded()

    a = 200
    b = -3   
    
    for score in user_satisfaction_list:  
        cost += a*math.exp(b*score) - a*math.exp(b)
    return cost



# ==========================================
# 黄金配置函数 2: Cost (安全约束)
# 目标：监测变压器过载 (硬约束)
# ==========================================
def paper_cost_function(env, total_costs, user_satisfaction_list, *args):
    """
    Cost = 仅包含变压器过载量 (kW)
    注意：这里绝对不要包含用户满意度！因为那不是'安全'事故。
    """
    cost = 0

    # 只计算物理层面的过载
    for tr in env.transformers:
        # get_how_overloaded() 返回的是过载的功率 (kW)
        overload = tr.get_how_overloaded()
        
        # 如果过载了，就计入 Cost
        # 建议系数为 1.0，这样 Cost=5 就代表过载了 5kW，物理意义清晰
        if overload > 0:
            cost += 1.0 * overload
            
    # [DEBUG] Print Cost
    # if cost > 0:
    #    print(f"[DEBUG] Step Cost: Overload={cost:.2f} kW")

    return cost



def ProfitMax_TrPenalty_UserIncentives_safety(env, total_costs, user_satisfaction_list, *args):
    
    user_penalty = usrpenalty_cost(env, total_costs, user_satisfaction_list)
    reward = total_costs    # 总奖励 = 钱
    # [DEBUG] Print components
    # print(f"[DEBUG] Step Reward: Profit={reward:.2f}, UserPenalty={user_penalty:.2f}, Final={reward - user_penalty:.2f}")
    return reward - user_penalty



def V2G_profitmaxV2(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    verbose = False
    
    if verbose:
        print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    
    linear = False
    if linear:
        cost_multiplier = 0.1
    else:
        cost_multiplier = 0.05
    
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                min_steps_to_full = (ev.desired_capacity - ev.current_capacity) / \
                    (ev.max_ac_charge_power/(60/env.timescale))
                
                
                departing_step = ev.time_of_departure - env.current_step
                
                cost = 0
                if min_steps_to_full > departing_step:                    
                    min_capacity_at_time = ev.desired_capacity - ((departing_step+1) * ev.max_ac_charge_power/(60/env.timescale))
                    
                    if linear:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)
                    else:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)**2
                        
                    if verbose:
                        print(f'min_capacity_at_time: {min_capacity_at_time} | {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}') 
                    user_costs += - cost
                
                if verbose:
                    print(f'- EV: {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}')
                
    for ev in env.departing_evs:
        if ev.desired_capacity > ev.current_capacity:            
            if verbose:
                print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity} | costs: {-cost_multiplier*(ev.desired_capacity - ev.current_capacity)**2}')
                
            if linear:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)
            else:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)**2
            
    if verbose:
        print(f'!!! User Satisfaction Penalty: {user_costs}')

    return (reward + user_costs)