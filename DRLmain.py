import torch
import numpy as np
from env import SchedulingEnv  # 从env中引入SchedulingEnv类
from DRLmodel import baseline_DQN, baselines, baseline_PPO
from utils import get_args, plot_SR_figure
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import utils as utils
'''
    定义DQN和对比方法调度过程
'''
R = False
# 生成各项参数
args = get_args()  # param_parser搜索



performance_lamda = np.zeros(args.Baseline_num)
performance_success = np.zeros(args.Baseline_num)
performance_util = np.zeros(args.Baseline_num)
performance_finishT = np.zeros(args.Baseline_num)
performance_cost = np.zeros(args.Baseline_num)
performance_profit = np.zeros(args.Baseline_num)

# 启动环境
env = SchedulingEnv(args)

# DQN agent
brainRL = baseline_DQN(env.actionNum, env.s_features,
                       0.0005491897779608287,
                       0.5159231858657606,
                        e_greedy = args.e_greedy,
                        replace_target_iter = args.replace_target_iter,
                        memory_size = args.memory_size,
                        batch_size = args.batch_size,
                        e_greedy_increment = args.e_greedy_increment)


# PPO agent
PPO = baseline_PPO(n_actions=env.actionNum, n_features=env.s_features)

# baselines
brainOthers = baselines(env.actionNum, env.CPtypes)
"""
    parm:
        env.actionNum:动作空间数量
        env.CPtypes:CP的类型
"""

global_step = 0
my_learn_step = 0
DQN_Reward_list = []
PPO_Reward_list = []
#loss_list = []

final_reward_random = [0]*args.Epoch
final_reward_RR = [0]*args.Epoch
final_reward_early = [0]*args.Epoch
final_reward_DQN = [0]*args.Epoch
final_reward_PPO = [0]*args.Epoch


for episode in range(args.Epoch):

    print('----------------------------Episode', episode, '----------------------------')
    job_c = 1
    performance_c = 0
    env.reset(args)
    performance_respTs = []
    # done = False

    rewards_list_random = []
    rewards_list_RR = []
    rewards_list_early = []
    rewards_list_DQN = []
    rewards_list_PPO = []

    while True:
        # baseline DQN
        global_step += 1

        #  获得任务属性
        finish, job_attrs = env.workload(job_c)

        #  获得虚拟机和环境状态
        DQN_state = env.getState(job_attrs, 4)
        PPO_state = env.getState(job_attrs, 5)

        #  选择虚拟机执行任务，获得reward并记录状态迁移（s,a,s',r）;当已经调度的任务个数满足要求，开始学习
        #  第一轮时并不执行该语句，
        if global_step != 1:
            brainRL.store_transition(last_state_DQN, last_action_DQN, last_reward_DQN, DQN_state)
            PPO.store_transition(last_state_PPO, last_action_PPO, last_reward_PPO, PPO_state)

        action_DQN = brainRL.choose_action(DQN_state)  # choose action
        # print('DQN:', action_DQN)
        reward_DQN = env.feedback(job_attrs, action_DQN, 4)
        rewards_list_DQN.append(reward_DQN)

        action_PPO = PPO.choose_action(PPO_state)  # choose action
        # print('PPO:', action_PPO)
        reward_PPO = env.feedback(job_attrs, action_PPO, 5)
        rewards_list_PPO.append(reward_PPO)

        if episode == 1:
            DQN_Reward_list.append(reward_DQN)
            PPO_Reward_list.append(reward_PPO)
        if (global_step > args.Dqn_start_learn) and (global_step % args.Dqn_learn_interval == 0):  # learn
            #loss = brainRL.learn()
            brainRL.learn()
            PPO.learn()


        last_state_DQN = DQN_state
        last_action_DQN = action_DQN
        last_reward_DQN = reward_DQN

        last_state_PPO = PPO_state
        last_action_PPO = action_PPO
        last_reward_PPO = reward_PPO

        """
            Baseline method
        """
        # random policy
        state_Ran = env.getState(job_attrs, 1)
        action_random = brainOthers.random_choose_action()
        reward_random = env.feedback(job_attrs, action_random, 1)
        rewards_list_random.append(reward_random)
        # round robin policy
        state_RR = env.getState(job_attrs, 2)
        action_RR = brainOthers.RR_choose_action(job_c)
        reward_RR = env.feedback(job_attrs, action_RR, 2)
        rewards_list_RR.append(reward_RR)
        # earliest policy
        idleTimes = env.get_CP_idleT(3)  # get CP state
        action_early = brainOthers.early_choose_action(idleTimes)
        reward_early = env.feedback(job_attrs, action_early, 3)
        rewards_list_early.append(reward_early)

        #  获取 500 次执行的性能指标总和
        if job_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(args.Baseline_num, performance_c, job_c)
            cost = env.get_accumulateCost(args.Baseline_num, performance_c, job_c)
            finishTs = env.get_FinishTimes(args.Baseline_num, performance_c, job_c)
            avg_exeTs = env.get_executeTs(args.Baseline_num, performance_c, job_c)
            avg_waitTs = env.get_waitTs(args.Baseline_num, performance_c, job_c)
            avg_respTs = env.get_responseTs(args.Baseline_num, performance_c, job_c)
            performance_respTs.append(avg_respTs)
            successTs = env.get_successTimes(args.Baseline_num, performance_c, job_c)
            performance_c = job_c

        job_c += 1
        if finish:
            break

    final_reward_random[episode] = sum(rewards_list_random)
    final_reward_RR[episode] = sum(rewards_list_RR)
    final_reward_early[episode] = sum(rewards_list_early)
    final_reward_DQN[episode] = sum(rewards_list_DQN)
    final_reward_PPO[episode] = sum(rewards_list_PPO)

    startP = 125

    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_total_responseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    avg_util = env.get_avgUtilitizationRate(args.Baseline_num, startP)
    total_Ts = env.get_totalTimes(args.Baseline_num, startP)
    total_cost = env.get_totalCost(args.Baseline_num, startP)
    total_profit = env.get_totalProfit(args.Baseline_num, startP)
    print('total performance (after 2000 jobs):')
    for i in range(len(args.Baselines)):
        name = "[" + args.Baselines[i] + "]"
        print(name + " reward:", total_Rewards[i], ' avg_responseT:', avg_allRespTs[i],
              'success_rate:', total_success[i], ' utilizationRate:', avg_util[i], ' finishT:', total_Ts[i], 'Cost:',
              total_cost[i],'profit:',total_profit[i])

    if episode != 0:
        performance_lamda[:] += env.get_total_responseTs(args.Baseline_num, 0)
        performance_success[:] += env.get_totalSuccess(args.Baseline_num, 0)
        performance_util[:] += env.get_avgUtilitizationRate(args.Baseline_num, 0)
        performance_finishT[:] += env.get_totalTimes(args.Baseline_num, 0)
        performance_cost += env.get_totalCost(args.Baseline_num, 0)
        performance_profit += env.get_totalProfit(args.Baseline_num, 0)

    # brainRL.plot_loss_history(episode)
    # PPO.plot_loss_history(episode)
print('')

print('---------------------------- Final results ----------------------------')
performance_lamda = np.around(performance_lamda / (args.Epoch - 1), 3)
performance_success = np.around(performance_success / (args.Epoch - 1), 3)
performance_util = np.around(performance_util / (args.Epoch - 1), 3)
performance_finishT = np.around(performance_finishT / (args.Epoch - 1), 3)
performance_cost = np.around(performance_cost / (args.Epoch - 1), 3)
performance_profit = np.around(performance_profit / (args.Epoch - 1), 3)
print('avg_responseT:')
print(performance_lamda)
print('success_rate:')
print(performance_success)
print('utilizationRate:')
print(performance_util)
print('finishT:')
print(performance_finishT)
print('Cost:')
print(performance_cost)
print('Profit:')
print(performance_profit)
