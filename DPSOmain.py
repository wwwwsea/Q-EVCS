import numpy as np
from matplotlib import pyplot as plt
import torch
import numpy as np
from env2 import SchedulingEnv  # 从env中引入SchedulingEnv类
from model import baseline_DQN, baselines, baseline_PPO
from utils import get_args, plot_SR_figure
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import utils as utils
from DPSOmodel import DPSO,DParticle,calculate_fitnessgb,calculate_fitness
import copy

args = get_args()  # param_parser搜索
if __name__ == '__main__':
    #for args.lamda in [10,20,30,40]:
    performance_lamda = 0
    performance_success = 0
    performance_util = 0
    performance_finishT = 0
    performance_cost = 0
    performance_profit = 0

    joblist=[]
    env = SchedulingEnv(args)

    for i in range(1,501):
        # print(i)
        finish, job_attrs = env.workload(i)
        joblist.append(job_attrs)
        # print(job_attrs[0],job_attrs[1])
        t=0
    jobtime = []
    j = 1
    l = 0
    t = 0  # 确保 t 有一个初始值，如果 t 从外部定义，请忽略这个初始化

    while len(joblist) != 0:
        j_t = False
        # 确保 jobtime 的长度足够
        if len(jobtime) < j:
            jobtime.append([])  # 添加一个新的空列表以供使用

        i = len(joblist) - 1
        while i >= 0:
            if joblist[i][1] >= t - 1 and joblist[i][1] <= t:
                jobtime[j - 1].append(joblist.pop(i))  # 从列表中移除元素
                j_t = True
            i -= 1  # 更新索引

        l += len(jobtime[j - 1])
        t += 0.25
        if j_t:
            j += 1  # 准备下一个时间段的列表

    # print("jobtime =", jobtime)
    # print("总任务数 l =", l)
    bestP = []
    CP = np.zeros((2, 10))
    event = np.zeros((10, 500))
    env = SchedulingEnv(args)
    jobl=[]
    for i in range(len(jobtime)):
        print("----",len(jobtime[i]))
        if(i==0):
            dpso = DPSO(jobtime[i],CP,event,env)
            result = dpso.execute()

            if isinstance(result, np.ndarray):
                bestP=result.tolist()
            else:
                bestP=result
            calculate_fitness(bestP, jobtime[i],CP,event, env)
            jobl=jobtime[i]
            # print(type(bestP))
            # print(bestP)

            CP = copy.deepcopy(env.getCP())
            event = copy.deepcopy(env.getevent())
            # print("((((",CP)
        else:
            # print("bestP----",type(bestP))
            dpso = DPSO(jobtime[i], CP, event,env)
            result = dpso.execute()

            if isinstance(result, np.ndarray):
                bestP.extend(result.tolist())
            else:
                bestP.extend(result)


            # print("0000,",bestP)
            # print(bestP[1])
            # print(jobl)
            # print(jobtime[i])

            jobl.extend(jobtime[i])
            # print(jobl)
            # print(jobl[2])
            CP_i=np.zeros((2, 10))
            event_i = np.zeros((10, 500))
            calculate_fitness(bestP, jobl,CP_i,event_i, env)
            CP = copy.deepcopy(env.getCP())
            event = copy.deepcopy(env.getevent())

    calculate_fitnessgb(bestP,jobl,env)

    startP = 0

    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_total_responseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    avg_util = env.get_avgUtilitizationRate(args.Baseline_num, startP)
    total_Ts = env.get_totalTimes(args.Baseline_num, startP)
    total_cost = env.get_totalCost(args.Baseline_num, startP)
    total_profit = env.get_totalProfit(args.Baseline_num, startP)


    name = "[" + args.Baselines[3] + "]"
    print(name + " reward:", total_Rewards[3], ' avg_responseT:', avg_allRespTs[3],
          'success_rate:', total_success[3], ' utilizationRate:', avg_util[3], ' finishT:', total_Ts[3],
          'Cost:',
          total_cost[3], 'profit:', total_profit[3])

    performance_lamda +=  avg_allRespTs[3]
    performance_success += total_success[3]
    performance_util += avg_util[3]
    performance_cost += total_cost[3]
    performance_profit += total_profit[3]










