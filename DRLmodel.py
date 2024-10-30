import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
random.seed(6)
np.random.seed(6)
from datetime import datetime

'''
    建立DQN、PPO模型
'''


class NET(nn.Module):
    def __init__(self, n_actions, n_features):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(10, 10)
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        q_evaluate = self.out(x)
        return q_evaluate


class baseline_DQN():
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.7,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=1500,
            batch_size=128,
            e_greedy_increment=0.006,
            output_graph=False,
    ):
        self.n_actions = n_actions  # if +1: allow to reject jobs
        self.n_features = n_features
        self.lr = learning_rate # 学习率
        self.gamma = reward_decay # 奖励折扣
        self.epsilon_max = e_greedy # c-greedy
        self.replace_target_iter = replace_target_iter # 目标更新
        self.memory_size = memory_size # 经验池大小
        self.batch_size = batch_size # 抽样大小
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.01 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0  # total learning step
        self.replay_buffer = deque()  # init experience replay [s, a, r, s_, done]

        # consist of [target_net, evaluate_net]
        self.eval_net, self.target_net = NET(n_actions, n_features), NET(n_actions, n_features)
        total_params = sum(p.numel() for p in self.eval_net.parameters() if p.requires_grad)
        print(f"Total trainable parameters in the DQN model: {total_params}")
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()  # 优化器和损失函数
        self.loss_history = []


    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        pro = np.random.uniform()
        if pro < self.epsilon:
            # 计算q——vals
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1]
            # print('pro: ', pro, ' q-values:', actions_value, '  best_action:', action)
            # print('  best_action:', action)
        else:
            action = np.random.randint(0, self.n_actions)
            # print('pro: ', pro, '  rand_action:', action)
            # print('  rand_action:', action)
        return action

    # def store_transition(self, s, a, r, s_):
    #     one_hot_action = np.zeros(self.n_actions)   #
    #     one_hot_action[a] = 1
    #     self.replay_buffer.append((s, one_hot_action, r, s_))
    #     if len(self.replay_buffer) > self.memory_size:
    #         self.replay_buffer.popleft()
    #
    # def learn(self):
    #     # check to replace target parameters
    #     if self.learn_step_counter % self.replace_target_iter == 0:  # 50
    #         self.target_net.load_state_dict(self.eval_net.state_dict())
    #         # print('-------------target_params_replaced------------------')
    #
    #     # sample batch memory from all memory: [s, a, r, s_]
    #     minibatch = random.sample(self.replay_buffer, self.batch_size)
    #     # 打包记忆，分开保存进b_s，b_a，b_r，b_s
    #     b_s = torch.FloatTensor([data[0] for data in minibatch])
    #     b_a = []
    #
    #     for data in minibatch:
    #         for i in range(0, len(data[1])):
    #             if (data[1][i] == 1):
    #                 b_a.append(i)
    #     b_a = torch.LongTensor(b_a)
    #     b_a = b_a.reshape(30, 1)
    #     b_r = torch.FloatTensor([data[2] for data in minibatch]).reshape(30, 1)
    #     b_s_ = torch.FloatTensor([data[3] for data in minibatch])
    #
    #     q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
    #     q_next = self.target_net(b_s_).detach()  # q_next不进行反向传递误差, so detach
    #     q_target = b_r + self.gamma * q_next.max(1)[0].reshape(30, 1)  # shape (batch, 1)
    #     loss = self.loss_func(q_eval, q_target)
    #     # 计算, 更新 eval net
    #     self.optimizer.zero_grad()
    #     loss.backward()  # 误差反向传播
    #     self.optimizer.step()
    #
    #     # increasing epsilon
    #     if self.epsilon < self.epsilon_max:
    #         self.epsilon = self.epsilon + self.epsilon_increment
    #     else:
    #         self.epsilon = self.epsilon_max
    #     # print('epsilon:', self.epsilon)
    #     self.learn_step_counter += 1

    def store_transition(self, s, a, r, s_):
        self.replay_buffer.append((s, a, r, s_))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:  # 50
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('-------------target_params_replaced------------------')

        # sample batch memory from all memory: [s, a, r, s_]
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # 从经验回放缓冲区 (replay_buffer) 中随机抽样一个批次 (minibatch) 的记忆
        # 其中每个记忆包含了状态 (s)、动作 (a)、奖励 (r) 和下一个状态 (s_)。
        b_s = torch.FloatTensor([data[0] for data in minibatch]) #状态
        b_a = torch.LongTensor([data[1] for data in minibatch])# 动作
        #print("before  ",b_a.shape)
        b_a = b_a.reshape(self.batch_size, 1)
        b_r = torch.FloatTensor([data[2] for data in minibatch]).reshape(self.batch_size, 1)
        b_s_ = torch.FloatTensor([data[3] for data in minibatch])# 下一个状态
        #print(b_s.shape,b_r.shape,b_a.shape,b_s_.shape)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + self.gamma * q_next.max(1)[0].reshape(self.batch_size, 1)  # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        self.loss_history.append(loss.item())

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()  # 误差反向传播
        self.optimizer.step()

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1
        return loss

    def getlossHist(self):
        return self.loss_history

    def plot_loss_history(self):
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.family'] = 'Times New Roman'

        plt.plot(self.loss_history)  # 绘制Loss值曲线
        import pickle


        print("-----------",self.loss_history)

        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Training Loss', fontsize=14)
        plt.tick_params(top=True, right=True, direction='in')
        plt.grid(True, linestyle='--', alpha=0.6)

        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        filename = f"E:/0SSTUDY-FIRST/CASA资料/(测试)CASA Cost-effective EV Charging Scheduling based on Deep Reinforcement Learning - 副本/data1/DQN_loss_{timestamp}.pdf"

        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=600, format='pdf')  # Adjusting bbox_inches
        print("saved",filename)
        plt.show()




class baselines:
    def __init__(self, n_actions, CPtypes):
        self.n_actions = n_actions
        self.CPtypes = np.array(CPtypes)  # change list to numpy
        # parameters for sensible policy
        self.sensible_updateT = 5
        self.sensible_counterT = 1
        self.sensible_discount = 0.7  # 0.7 is best, 0.5 and 0.6 OK
        self.sensible_W = np.zeros(self.n_actions)
        self.sensible_probs = np.ones(self.n_actions) / self.n_actions
        self.sensible_probsCumsum = self.sensible_probs.cumsum()
        self.sensible_sumDurations = np.zeros((2, self.n_actions))  # row 1: jobNum   row 2: sum duration

    def random_choose_action(self):  # random policy
        action = np.random.randint(self.n_actions)  # [0, n_actions)
        return action

    def RR_choose_action(self, job_count):  # round robin policy
        action = (job_count - 1) % self.n_actions
        return action

    def early_choose_action(self, idleTs):  # earliest policy
        action = np.argmin(idleTs)
        return action


"""
    
    GPT-PPO

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=10):
        super(ActorCritic, self).__init__()

        # Actor Network，输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network，输出状态值估计
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        # 返回动作概率分布和状态值估计
        return self.actor(state), self.critic(state)


class baseline_PPO:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.007919303729010148,
            gamma=0.6678709237403223,
            epsilon_clip=0.1370183043585458,
            value_loss_coeff=0.48912782410954764,
            entropy_coeff=0.02411360491766849,
            memory_size=1500,
            batch_size=30,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_actor = learning_rate
        self.lr_critic = learning_rate
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.loss_history = []

        # Actor-Critic网络
        self.actor_critic = ActorCritic(n_features, n_actions)

        total_params = sum(p.numel() for p in self.actor_critic.parameters() if p.requires_grad)
        print(f"Total trainable parameters in the PPO model: {total_params}")

        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=self.lr_critic)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor_critic.actor(state)

        # 检查 action_probs 是否包含 NaN
        if torch.isnan(action_probs).any():
            print("NaN detected in action probabilities.")
            return -1  # 返回一个特殊值来表示 NaN 问题

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state):
        #print('self.memory',len(self.memory))
        self.memory.append((state, action, reward, next_state))
        #print("self.memory_size",self.memory_size)
        if len(self.memory) > self.memory_size:
            #print("@@@@@")
            self.memory.pop(0)

    def learn(self):
        #print(len(self.memory))
        if len(self.memory) < self.batch_size:
            return

        batch_indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        #print("1")

        # 计算状态值估计
        values = self.actor_critic.critic(states)

        # 计算优势函数
        advantages = rewards - values.squeeze().detach()

        # 计算动作概率分布
        action_probs = self.actor_critic.actor(states)
        old_action_probs = action_probs[range(len(action_probs)), actions]

        # 计算重要性采样比率
        action_dist = torch.distributions.Categorical(action_probs)
        new_action_probs = action_dist.log_prob(actions)
        ratio = torch.exp(new_action_probs - old_action_probs)

        # 计算PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # # 计算Critic损失
        # critic_loss = nn.MSELoss()(values, rewards)
        # 计算Critic损失
        critic_loss = nn.MSELoss()(values.squeeze(), rewards)

        # 计算熵正则化项
        entropy = -torch.sum(action_probs * torch.log(action_probs), dim=1).mean()

        # 总损失
        total_loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy
        self.loss_history.append(total_loss.item())
        #print(len(self.loss_history))

        # 更新Actor和Critic网络
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        # 清空记忆
        self.memory = []
    def getlossHist(self):
        print("self.loss_history",self.loss_history)
        return self.loss_history
    def plot_loss_history(self, episode):

        plt.rcParams['font.family'] = 'Times New Roman'

        plt.plot(self.loss_history)  # 绘制Loss值曲线
        plt.xlabel('Iterations', fontsize=14)  # X轴标签
        plt.ylabel('Training Loss', fontsize=14)  # Y轴标签
        plt.tick_params(top=True, right=True, direction='in')
        plt.grid(True, linestyle='--', alpha=0.6)

        filename = f'PPO_loss_episode_{episode}.pdf'
        plt.savefig(filename, dpi=600, format='pdf')

        plt.show()  # 显示图像




    # class NET(nn.Module):
    #     def __init__(self, n_actions, n_features):
    #         super(NET, self).__init__()
    #         self.fc1 = nn.Linear(n_features, 64)
    #         self.fc2 = nn.Linear(64, 64)
    #         self.fc3 = nn.Linear(64, 64)
    #         self.out = nn.Linear(64, n_actions)
    #
    #     def forward(self, x):
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = F.relu(self.fc3(x))
    #         q_values = self.out(x)
    #         return q_values


    # # def choose_action(self, state):
    # #     state = torch.FloatTensor(state)
    # #     if np.isnan(state).any():
    # #         raise ValueError("There is a NaN value")
    # #
    # #     probs, _ = self.policy_net(state)
    # #
    # #     # print("Action Probabilities:", probs.detach().numpy())
    # #     action = np.random.choice(range(self.n_actions), p=probs.detach().numpy())
    # #     return action
    #
    # def choose_action(self, state):
    #     state = torch.FloatTensor(state)
    #     if np.isnan(state).any():
    #         raise ValueError("There is a NaN value")
    #
    #     logits, _ = self.policy_net(state)  # 使用策略网络policy_net计算状态state对应的策略网络输出
    #
    #     # 数值稳定性处理
    #     probabilities = F.softmax(logits, dim=-1)  # 对logits进行softmax操作，将其转换为动作的概率分布
    #     epsilon = 1e-10
    #     probabilities = torch.clamp(probabilities, min=epsilon,
    #                                 max=1.0 - epsilon)
    #     print("2.数值稳定：", probabilities)
    #
    #     probabilities /= probabilities.sum()
    #     print("3.归一化：", probabilities)
    #
    #     action = 0
    #
    #     # range(self.n_actions)
    #     # p=probabilities.detach().numpy() —— 张量probabilities包含了每个动作的概率分布，detach().numpy()方法将张量转换为NumPy数组
    #     #                                  —— 作为np.random.choice函数的一个参数，它指定了每个选项被选择的概率。
    #     # np.random.choice
    #     if not (np.isnan(probabilities.detach().numpy()).any()):
    #         action = np.random.choice(range(self.n_actions), p=probabilities.detach().numpy())
    #     return action
    #
    # # def choose_action(self, state):
    # #     state = torch.FloatTensor(state)
    # #     logits, _ = self.policy_net(state)
    # #     # print(logits)
    #
    #     # plan B
    #     # action = torch.multinomial(logits, 1).item()
    #     #
    #     #
    #     # # max_action = logits.argmax().item()
    #     # # result = action == max_action
    #     # # print(result)
    #     #
    #     # return action
    #
    #     # # plan A
    #     # #  对logits进行softmax操作，将其转换为动作的概率分布
    #     # #  softmax 函数将每个类别的得分映射到0到1之间，并确保所有类别的概率之和为1。
    #     # #  dim=-1 表示在最后一个维度上执行softmax
    #     # probabilities = F.softmax(logits, dim=-1)
    #     # print("生成概率：", probabilities)
    #     #
    #     # epsilon = 1e-10
    #     # probabilities = torch.clamp(probabilities, min=epsilon, max=1.0 - epsilon)
    #     #
    #     # # # range(self.n_actions)
    #     # # # p=probabilities.detach().numpy()
    #     # # # np.random.choice
    #     # # action = np.random.choice(range(self.n_actions), p=probabilities.detach().numpy())
    #     # # return action
    #     #
    #     # if not (np.isnan(probabilities.detach().numpy()).any()):
    #     #     action = np.random.choice(range(self.n_actions), p=probabilities.detach().numpy())
    #     # return action
    #
    #
    # def store_transition(self, s, a, r, s_):
    #     # print("State:", s)
    #     # print("Action:", a)
    #     # print("Reward:", r)
    #     # print("Next State:", s_)
    #     self.memory.append((s, a, r, s_))
    #     self.memory_counter += 1
    #
    #
    # def learn(self):
    #     # if self.memory_counter < self.memory_size:
    #     #     return
    #
    #     if self.memory_counter < 200:
    #         return
    #
    #     for i in range(100):
    #         i = i+1
    #         mini_batch = random.sample(self.memory, self.batch_size)
    #
    #         s, a, r, s_ = zip(*mini_batch)
    #         s = torch.FloatTensor(s)
    #         a = torch.LongTensor(a).view(-1, 1)
    #         r = torch.FloatTensor(r).view(-1, 1)
    #         s_ = torch.FloatTensor(s_)
    #
    #         # 计算old_probs和old_values
    #         old_probs, old_values = self.policy_net(s)
    #         # 计算new_values
    #         _, new_values = self.policy_net(s_)
    #
    #         advantages = r + self.gamma * new_values - old_values
    #
    #         # 计算ratio，用于clip
    #         action_probs = old_probs.gather(1, a)
    #         ratio = action_probs / (old_probs.detach())  # 计算一个比率ratio，用于执行PPO中的策略更新。
    #                                                      # .detach()用于分离old_probs，防止梯度传播。
    #
    #         # 计算actor loss
    #         surr1 = ratio * advantages
    #         surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
    #         actor_loss = -torch.min(surr1, surr2).mean()  # 计算演员（actor）损失，它是策略梯度的损失函数
    #
    #         # 计算critic loss
    #         critic_loss = F.mse_loss(new_values, r)       # 计算评论家（critic）损失，它是值网络的均方误差损失，用于优化值函数。
    #
    #         # 计算entropy loss，用于增加探索
    #         entropy = -(old_probs * torch.log(old_probs + 1e-10)).sum(dim=1).mean()  # 计算策略熵损失，用于增加探索性
    #
    #         # 计算总损失，包括上述损失的总和
    #         total_loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy
    #
    #         self.optimizer.zero_grad()
    #         total_loss.backward()  # 使用反向传播计算损失函数关于模型参数的梯度
    #
    #         self.optimizer.step()



"""
    DDPG_None
"""
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# from collections import deque
#
# class baseline_DDPG():
#     def __init__(
#         self,
#         n_actions,
#         n_features,
#         actor_lr=0.001,
#         critic_lr=0.001,
#         gamma=0.99,
#         memory_capacity=10000,
#         batch_size=64,
#         tau=0.001,
#         # actor and critic network architectures
#     ):
#         self.n_actions = n_actions
#         self.n_features = n_features
#         self.gamma = gamma
#         self.memory_capacity = memory_capacity
#         self.batch_size = batch_size
#         self.tau = tau
#
#         # Actor network
#         self.actor = ActorNet(n_features, n_actions)
#         self.actor_target = ActorNet(n_features, n_actions)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
#
#         # Critic network
#         self.critic = CriticNet(n_features, n_actions)
#         self.critic_target = CriticNet(n_features, n_actions)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
#
#         # Experience replay buffer
#         self.memory = deque(maxlen=memory_capacity)
#
#     # def choose_action(self, state):
#     #     # state = torch.FloatTensor(state)
#     #     # action = self.actor(state).detach().numpy()
#     #     # return action
#     #
#     #     state = torch.FloatTensor(state)
#     #     action = self.actor(state).detach().numpy()
#     #     # action = np.argmax(action_probs)
#     #
#     #     # print('DDPG prob:', action_probs)
#     #     # print('DDPG action:', action)
#     #
#     #     return action
#
#     def choose_action(self, state):
#         state = torch.FloatTensor(state)
#         action = self.actor(state).detach().numpy()
#         return action
#
#     def store_transition(self, s, a, r, s_):
#         self.memory.append((s, a, r, s_))
#
#     def learn(self):
#         if len(self.memory) < self.batch_size:
#             return
#
#         # Sample a batch of experiences from the replay buffer
#         minibatch = random.sample(self.memory, self.batch_size)
#         s_batch, a_batch, r_batch, s_next_batch = zip(*minibatch)
#         s_batch = torch.FloatTensor(s_batch)
#         a_batch = torch.FloatTensor(a_batch)
#         r_batch = torch.FloatTensor(r_batch)
#         s_next_batch = torch.FloatTensor(s_next_batch)
#
#         # Update critic
#         q_values = self.critic(s_batch, a_batch)
#         next_actions = self.actor_target(s_next_batch).detach()
#         target_q_values = r_batch + self.gamma * self.critic_target(s_next_batch, next_actions).detach()
#         critic_loss = nn.MSELoss()(q_values, target_q_values)
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()
#
#         # Update actor
#         policy_loss = -self.critic(s_batch, self.actor(s_batch)).mean()
#         self.actor_optimizer.zero_grad()
#         policy_loss.backward()
#         self.actor_optimizer.step()
#
#         # Soft update target networks
#         self.soft_update_target_networks()
#
#     def soft_update_target_networks(self):
#         for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
#             target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
#         for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
#             target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
#
# # Define actor network architecture
# class ActorNet(nn.Module):
#     def __init__(self, n_features, n_actions):
#         super(ActorNet, self).__init__()
#         self.fc1 = nn.Linear(n_features, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, n_actions)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.tanh(self.output(x))
#         return x
#
# # Define critic network architecture
# class CriticNet(nn.Module):
#     def __init__(self, n_features, n_actions):
#         super(CriticNet, self).__init__()
#         self.fc1 = nn.Linear(n_features + n_actions, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, 1)
#
#     def forward(self, state, action):
#         x = torch.cat((state, action), dim=1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.output(x)
#         return x
#
# # Usage example:
# # ddpg = baseline_DDPG(n_actions=2, n_features=4)
# # for episode in range(1000):
# #     state = env.reset()
# #     for step in range(200):
# #         action = ddpg.choose_action(state)
# #         next_state, reward, done, _ = env.step(action)
# #         ddpg.store_transition(state, action, reward, next_state)
# #         state = next_state
# #         ddpg.learn()





"""
    PPO-NaN error
"""
# def choose_action(self, state):
#     state = torch.FloatTensor(state)
#     if np.isnan(state).any():  # 检查，如果状态中包含NaN值，它会引发ValueError异常。
#         raise ValueError("There is a NaN value")
#
#     logits, _ = self.policy_net(state)
#
#     # 数值稳定性处理
#     probabilities = F.softmax(logits, dim=-1)
#     epsilon = 1e-10
#     probabilities = torch.clamp(probabilities, min=epsilon,
#                                 max=1.0 - epsilon)
#     print("2.数值稳定：", probabilities)
#
#     # 重新归一化概率分布
#     probabilities /= probabilities.sum()
#     print("3.归一化：", probabilities)
#
#     action = 0
#
#     # range(self.n_actions）
#     # p=probabilities.detach().numpy() —— 张量probabilities包含了每个动作的概率分布，detach().numpy()方法将张量转换为NumPy数组
#     #                                  —— 作为np.random.choice函数的一个参数，它指定了每个选项被选择的概率。
#     # np.random.choice

#     if not (np.isnan(probabilities.detach().numpy()).any()):
#         action = np.random.choice(range(self.n_actions), p=probabilities.detach().numpy())
#     return action


