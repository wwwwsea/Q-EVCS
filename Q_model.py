import tensorflow as tf
import tensorflow_quantum as tfq
from collections import deque
import subprocess
import numpy as np
from env import SchedulingEnv
from cirq.contrib.svg import SVGCircuit
import cirq, sympy
from tensorflow.keras.utils import plot_model

tf.get_logger().setLevel('ERROR')


def get_gpu_info():
    try:
        return subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    except Exception as e:
        return str(e)


# 基本酉
def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

# 量子纠缠CNOT
def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits`
    (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

# 生成量子电路
def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3 * (n_layers + 1) * n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i, q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)

# 重上传电路
class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])

    def get_config(self):
        config = super(ReUploadingPQC, self).get_config()
        config.update({
            'qubits': self.n_qubits,
            'n_layers': self.n_layers
        })
        return config

# 可训练缩放权重
class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1, input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs + 1) / 2, tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0))

"""
n_qubits 环境中状态向量的维度。
n_layers 参数化量子电路（PQC）中的层数。
n_actions  环境中可用的动作数量。
"""
# 量子模型
def generate_model_Qlearning(n_qubits, n_layers, n_actions, target):
    """
    Generates a Keras model for a data re-uploading PQC Q-function approximator.
    作为Q值近似器
    """

    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [ops[0] * ops[1], ops[1] * ops[2], ops[2] * ops[3], ops[3] * ops[4], ops[4] * ops[5], ops[5] * ops[6],
                   ops[6] * ops[7], ops[7] * ops[8], ops[8] * ops[9], ops[9] * ops[
                       10]]  # observables = [ops[i] * ops[i + 1] for i in range(n_qubits - 1)] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
    input_tensor = tf.keras.Input(shape=(len(qubits),), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target * "Target" + "Q-values")
    # 这里的Rescaling返回的对象是一个层而非张量
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model


optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
# Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2





@tf.function
def Q_learning_update(states, actions, rewards, next_states, model, model_target, gamma, n_actions, opt_in, opt_var,
                      opt_out):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    rewards = tf.convert_to_tensor(rewards)
    next_states = tf.convert_to_tensor(next_states)

    # Compute their target q_values and the masks on sampled actions
    Q_target = model_target([next_states])
    target_q_values = rewards + (gamma * tf.reduce_max(Q_target, axis=1))
    masks = tf.one_hot(actions, n_actions)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([states])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

    return loss



def generate_memory(env, TrainJobNum, TrainArriveRate, TrainJobType, args):
    env.reset(args, TrainJobNum, TrainArriveRate, TrainJobType)
    replay_memory = deque(maxlen=15000)
    EV_c = 1
    step_count = 0
    print("*******************generate_Dataset*******************")
    while True:
        step_count += 1
        # 包括任务的索引、到达时间、执行时间、类型和截止时间。
        finish, job_attrs = env.workload(EV_c)
        # 选择5 使用 QRL 策略
        # 这是一个表示当前任务状态的向量。它包含任务类型（job_type）和相对于每个充电桩的等待时间
        QDQN_state = env.getState(job_attrs, 5)
        # 不是第一步就加入回放池
        if step_count != 1:
            replay_memory.append({'state': last_state,
                                  'action': last_action,
                                  'next_state': QDQN_state,
                                  'reward': float(last_reward)})
        # 随机选择一个动作作为当前动作
        action_QDQN = np.random.choice(args.n_actions)
        # 计算当前动作对应的奖励
        reward_QDQN = env.feedback(job_attrs, action_QDQN, 5)
        # st,at,rt
        last_state = QDQN_state
        last_action = action_QDQN
        last_reward = reward_QDQN

        EV_c += 1
        if EV_c % 1000 == 0:
            print(EV_c)
        if finish:
            return replay_memory




def QDQN_test(model, env, args, job_num, lamda, job_type):
    # 用于记录每个测试 episode 的累计奖励
    episode_reward_history = []
    #  用于记录每个时间步所选择的动作
    action_list = []
    # 记录整个测试过程的总奖励。
    total_reward = 0
    print("*******************testing*******************")
    job_c = 1
    #  重置环境，设置测试的作业数量、到达率、作业类型。
    env.reset(args, job_num, lamda, job_type)
    while True:
        #获取当前作业的属性，包括是否完成和其他信息。
        finish, job_attrs = env.workload(job_c)
        # 根据当前作业属性获取 Q-learning 模型的状态。
        QDQN_state = env.getState(job_attrs, 5)
        # 获取 Q-values。
        q_vals = model([tf.convert_to_tensor([QDQN_state])])
        # 选择 Q-values 最大的动作。
        action_QDQN = int(tf.argmax(q_vals[0]).numpy())

        reward_QDQN = env.feedback(job_attrs, action_QDQN, 5)

        last_reward = reward_QDQN
        total_reward += last_reward
        action_list.append(action_QDQN)
        episode_reward_history.append(total_reward)

        job_c += 1
        if job_c % 500 == 0:
            print(job_c)
        if finish:
            # print(action_list)
            total_success = env.get_totalSuccess(args.Baseline_num, 1)
            avg_allRespTs = env.get_total_responseTs(args.Baseline_num, 1)
            avg_util = env.get_avgUtilitizationRate(args.Baseline_num, 1)
            total_cost = env.get_totalCost(args.Baseline_num, 1)
            total_profit = env.get_totalProfit(args.Baseline_num, 1)
            print('responseT:', avg_allRespTs[4])
            print('success_rate:', total_success[4])
            print('utilizationRate:', avg_util[4])
            print('cost:', total_cost[4])
            print('profit:', total_profit[4])
            # frequency_counter = Counter(action_list)

            # # 准备数据
            # labels = list(range(10))
            # counts = [frequency_counter.get(i, 0) for i in labels]
            #
            # # 绘图
            # plt.figure(figsize=(10, 6))
            # plt.bar(labels, counts, color='blue', alpha=0.7, label='Frequency')
            # plt.xlabel('Charger ID')
            # plt.ylabel('Dispatch Frequency')
            # plt.title('Dispatch Frequency of Charging Station')
            # plt.xticks(labels)
            #
            # #plt.yticks(range(max(counts) + 1))
            # max_value = max(counts)
            # number_of_ticks = 10  # 希望的刻度数量
            # interval = max(1, int(max_value / number_of_ticks))  # 计算间隔，至少为1
            # plt.yticks(range(0, max_value + 1, interval))
            #
            # plt.legend()
            # plt.show()



            # 返回平均响应时间、成功率、利用率、成本。
            return avg_allRespTs[4], total_success[4], avg_util[4], total_cost[4],total_profit[4]

