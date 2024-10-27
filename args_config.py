from collections import namedtuple

Args = namedtuple('Args', [
    'Baselines', 'Baseline_num', 'Epoch',
    'Dqn_start_learn', 'Dqn_learn_interval',
    'epsilon_increment',
    'epsilon',
    'epsilon_max',
    'Lr_DDQN',
    'CP_Type', 'CP_Cost', 'CP_Acc', 'CP_Num', 'CP_capacity',
    'lamda', 'Job_Type', 'Job_Num', 'Job_len_Mean', 'Job_len_Std', 'Job_ddl',
    'n_qubits', 'n_layers', 'n_actions',
    'opt_in_rate', 'opt_var_rate', 'opt_out_rate',
    'batch_size', 'value_update', 'target_update', 'gamma'
])

# Create an Args object and provide default values for each field
args = Args(
    Baselines=['Random', 'Round-Robin', 'Earliest', 'DQN','QDQN'],
    Baseline_num=5,
    Epoch=100,
    Dqn_start_learn=500,
    Dqn_learn_interval=1,
    epsilon_increment = 0.06,
    epsilon = 0.01 ,
    epsilon_max = 0.9,
    Lr_DDQN=0.001,

    # CP_Type=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #0.1
    # CP_Type=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 0.3
    CP_Type=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],# 0.5
    # CP_Type=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 0.7
    # CP_Type=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 0.7
    #
    CP_Cost=[1, 1, 4, 4, 6, 1, 1, 4, 4, 6],
    CP_Acc=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2],
    CP_Num=10,
    CP_capacity=1000,
    lamda=20,
    Job_Type=0.5,
    Job_Num=500,
    Job_len_Mean=200,
    Job_len_Std=20,
    Job_ddl=0.25,

    n_qubits=11,  # Dimension of the state vectors in cloud task scheduling
    n_layers=8,  # Number of layers in the PQC
    n_actions=10,  # Number of virtual machines to which tasks can be assigned

    opt_in_rate=0.001,
    opt_var_rate=0.001,
    opt_out_rate=0.1,

    batch_size=32,
    value_update=1,  # Train the model every x steps
    target_update=30,  # Update the target model every x steps
    gamma=0.95  # Q-decay
)
