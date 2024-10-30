#!/usr/bin/env python
# coding: utf-8


# In[21]:

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)


from args_config import args
import numpy as np
import time
import csv
import os
from datetime import datetime
from env import SchedulingEnv
from Q_model import generate_model_Qlearning, generate_memory, QDQN_test, get_gpu_info
tf.get_logger().setLevel('ERROR')

from collections import deque, Counter



# In[22]:


env = SchedulingEnv(args)
gpu_info = get_gpu_info()

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



if __name__ == '__main__':


    gamma = 0.5  # Q-decay

    layers_num = 5
    iteration = 7000
    target_update = 3  # steps for update target-network
    batch_size = 128
    

    opt_in = 0.002
    opt_var = 0.02
    opt_out = 0.1

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=opt_in, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=opt_var, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=opt_out, amsgrad=True)
    w_in, w_var, w_out = 1, 0, 2

    model = generate_model_Qlearning(args.n_qubits, layers_num, args.n_actions, False)
    model_target = generate_model_Qlearning(args.n_qubits, layers_num, args.n_actions, True)  # tf.keras.Model
    model_target.set_weights(model.get_weights())  # 同样的初始化权重

    new_lamda = 20
    TrainJobNum = 5000
    TrainArriveRate = new_lamda
    TrainJobType = 0.5
    replay_memory = generate_memory(env, TrainJobNum, TrainArriveRate, TrainJobType, args)

    TrainJobNum2 = 5000
    TrainArriveRate2 = new_lamda
    replay_memory2 = generate_memory(env, TrainJobNum2, TrainArriveRate2, TrainJobType, args)

    for item in replay_memory2:
        replay_memory.append(item)

    TrainJobNum2 = 5000
    TrainArriveRate2 = new_lamda
    replay_memory2 = generate_memory(env, TrainJobNum2, TrainArriveRate2, TrainJobType, args)

    for item in replay_memory2:
        replay_memory.append(item)

    print(len(replay_memory))

    loss_list = []


    epoch = 1
    while True:
        #print(epoch)
        training_batch = np.random.choice(replay_memory, size=batch_size)
        loss = Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                 np.asarray([x['action'] for x in training_batch]),
                                 np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                 np.asarray([x['next_state'] for x in training_batch]),
                                 model=model, model_target=model_target,
                                 gamma=gamma,
                                 n_actions=args.n_actions,
                                 opt_in=optimizer_in,
                                 opt_var=optimizer_var,
                                 opt_out=optimizer_out)

        combined_tensor = tf.stack(loss)

        sum_value = tf.reduce_sum(combined_tensor)


        sum_value_numpy = sum_value.numpy()
        loss_list.append(sum_value_numpy)

        # Update target model
        if epoch % target_update == 0:
            model_target.set_weights(model.get_weights())

        epoch += 1
        if epoch % 20 == 0:
            loss_mean = round(np.mean(loss_list[-20:]), 4)
            loss_var = round(np.var(loss_list[-20:]), 4)
            print("last[",epoch,"]iterations loss mean:",loss_mean,"var:",loss_var)


        if  epoch == iteration:
            break

    TestJobNum = 5000
    TestArriveRate = 20
    TestJobType = 0.5
    print("TestArriveRate :", TestArriveRate)
    responseT, successRate, utiRate, cost, profit = QDQN_test(model, env, args, TestJobNum, TestArriveRate, TestJobType)




