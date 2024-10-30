import argparse

"""
    集中定义参数，方便调整
"""

def parameter_parser():

    parser = argparse.ArgumentParser(description="SAIRL")

    """
    add_argument()
    
        name or flags - 一个命名或者一个选项字符串的列表
        
        type - 为参数类型，例如int
        
        default - 当参数未在命令行中出现时使用的值

        choices - 用来选择输入参数的范围。例如choice = [1, 5, 10], 表示输入参数只能为1,5 或10
        
        help - 用来描述这个选项的作用
    """

    # 对比方法
    parser.add_argument("--Baselines",
                        type=list,
                        default=['Random', 'Round-Robin', 'Earliest', 'DQN', 'DDPG'],
                        help="Experiment Baseline")
    parser.add_argument("--Baseline_num",
                        type=int,
                        default=5,
                        help="Number of baselines")

    # 训练周期
    parser.add_argument("--Epoch",
                        type=int,
                        # default=40, # Initial
                        default=100,
                        help="Training Epochs")

    # 学习开始轮次
    parser.add_argument("--Dqn_start_learn",
                        type=int,
                        # default=500, # Initial
                        default=500,
                        help="Iteration start Learn for normal dqn")

    # 学习频率
    parser.add_argument("--Dqn_learn_interval",
                        type=int,
                        default=1,
                        help="Dqn's learning interval")

    # 充电桩类型： AC-II or DCFC
    parser.add_argument("--CP_Type",
                        type=list,
                        # default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #0.1
                        # default=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 0.3
                        # default=[ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],# 0.5
                        # default=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 0.7
                        # default=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.9
                        # default=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #0.1
                        # default=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 0.3
                        default=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],# 0.5
                        # default=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 0.7
                        # default=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 0.9
                        help="CP Type")
    #   成本
    parser.add_argument("--CP_Cost",
                        type=list,
                        # default=[1, 1, 2, 2, 4, 1, 1, 2, 2, 4],
                        default=[1, 1, 4, 4, 6, 1, 1, 4, 4, 6],
                        help="CP Cost")
    #   VCPU数量
    parser.add_argument("--CP_Acc",
                        type=list,
                        default=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2],
                        help="CP Cpus")
    #   虚拟机数量
    parser.add_argument("--CP_Num",
                        type=int,
                        default=10,
                        help="The number of CPs")
    #   虚拟机计算能力
    parser.add_argument("--CP_capacity",
                        type=int,
                        default=1000,
                        help="CP capacity")

    # 任务参数
    #   平均到达速度
    parser.add_argument("--lamda",
                        type=int,
                        default=20,
                        help="The parameter used to control the length of each jobs.")
    #   类型比例 IO sensitive or Computing sensitive
    parser.add_argument("--Job_Type",
                        type=float,
                        default=0.5,
                        help="The parameter used to control the type of each jobs.")

    # "learning_rate": learning_rate
    # 'reward_decay': reward_decay,
    # 'e_greedy': e_greedy,
    # 'replace_target_iter': replace_target_iter,
    # 'memory_size': memory_size,
    # 'batch_size': batch_size,
    # 'e_greedy_increment': e_greedy_increment,


    parser.add_argument("--learning_rate",
                        type=float,
                        #default=0.008815604018280842,
                        default=0.004987129546404974,
                        help="Learning rate for the agent.")

    parser.add_argument("--reward_decay",
                        type=float,
                        #default= 0.4392089858149745,
                        default= 0.6473765991240088,
                        help="Discount factor for future rewards.")

    parser.add_argument("--e_greedy",
                        type=float,
                        default=0.9,
                        help="Epsilon-greedy parameter for exploration vs exploitation.")

    parser.add_argument("--replace_target_iter",
                        type=int,
                        default=60,
                        help="Interval for replacing target network parameters.")

    parser.add_argument("--memory_size",
                        type=int,
                        default=1500,
                        help="Size of the replay memory buffer.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=96,
                        help="Batch size used for training the neural network.")

    parser.add_argument("--e_greedy_increment",
                        type=float,
                        default=0.006,
                        help="Increment for epsilon-greedy parameter over time.")


    #   数量
    parser.add_argument("--Job_Num",
                        type=int,
                        default=500,
                        # default=500,
                        help="The number of jobs.")
    #   正态分布均值平均计算量
    parser.add_argument("--Job_len_Mean",
                        type=int,
                        default=200,
                        help="The mean value of the normal distribution.")
    #   计算量标准差
    parser.add_argument("--Job_len_Std",
                        type=int,
                        default=20,
                        help="The std value of the normal distribution.")
    #   QoS 响应时间要求
    parser.add_argument("--Job_ddl",
                        type=float,
                        default=0.25,
                        help="Deadline time of each jobs")

    return parser.parse_args()

    # Plot
    # parser.add_argument("--Plot_labels",
    #                     type=list,
    #                     default=['b-', 'm-', 'g-', 'y-', 'r-', 'k-', 'w-'],
    #                     help="Deadline time of each jobs")


    # # DDQN学习率
    # parser.add_argument("--Lr_DDQN",
    #                     type=float,
    #                     default=0.001,
    #                     help="Dueling DQN Lr")