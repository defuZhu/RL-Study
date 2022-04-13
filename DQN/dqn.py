import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from cartpole_model import *
import numpy as np


class DQN(object):
    def __init__(self, model, gamma=None, lr=None):
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        # model
        self.eval_model = model
        self.target_model = copy.deepcopy(model)

        # use gpu
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.eval_model.to(device)
        # self.target_model.to(device)

        self.gamma = gamma
        self.lr = lr

        # loss function optimizer
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=self.lr)

    def predict(self, obs):
        """
        use self.eval_mode to predict the action value
        :param obs:
        :return:
        """
        pred_q = self.eval_model(obs)
        return pred_q

    def learn(self, obs, action, reward, next_obs, terminal):
        # turn np type into tensor with gradient attribute
        obs = Variable(torch.FloatTensor(obs))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        next_obs = Variable(torch.FloatTensor(next_obs))
        terminal = Variable(torch.LongTensor(terminal))

        # add a dimension
        action = torch.unsqueeze(action, 1)
        terminal = torch.unsqueeze(terminal, 1)
        reward = torch.unsqueeze(reward, 1)

        pred_value = self.eval_model(obs).gather(1, action)

        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)[0]  # max --> return tensor(data=[],indices=[])  max()[0] --> data
            target = reward + (1 - terminal) * self.gamma * max_v

        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())

#
# # super parameters
# BATCH_SIZE = 32  # batch size of sampling process from buffer
# LR = 0.01  # learning rate
# EPSILON = 0.9  # epsilon used for epsilon greedy approach
# GAMMA = 0.9  # discount factor
# TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target netowrk updates
# MEMORY_CAPACITY = 2000  # The capacity of experience replay buffer
#
#
# # 3. Define the DQN network and its corresponding methods
# class DQN(object):
#     def __init__(self, states_num, actions_num, ENV_A_SHAPE):
#         # -----------Define 2 networks (target and training)------#
#         self.states_num, self.actions_num = states_num, actions_num
#
#         # 两个相同的网络模型
#         self.eval_net, self.target_net = Net(self.states_num, self.actions_num), Net(self.states_num, self.actions_num)
#         # Define counter, memory size and loss function
#         self.learn_step_counter = 0  # count the steps of learning process
#         self.memory_counter = 0  # counter used for experience replay buffer
#
#         # ----Define the memory (or the buffer), allocate some space to it. The number
#         # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#  s有4个元素，两个s有 2*4个元素，a,r各一个元素 ： 2*4+2
#         self.memory = np.zeros((MEMORY_CAPACITY, self.states_num * 2 + 2))
#
#         # ------- Define the optimizer------#
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
#
#         # ------Define the loss function-----#
#         self.loss_func = nn.MSELoss()
#
#         self.ENV_A_SHAPE = ENV_A_SHAPE
#
#     def predict(self, obs):
#         x = torch.unsqueeze(torch.FloatTensor(obs), 0)
#         action = self.eval_net(x).detach()
#         action = torch.max(action, 1)[1].data.numpy()
#         action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
#         return action
#
#     def choose_action(self, x):
#         # This function is used to make decision based upon epsilon greedy
#         # x 表示状态s, 有4个元素
#         x = torch.unsqueeze(torch.FloatTensor(x), 0)  # add 1 dimension to input state x
#         # input only one sample
#         if np.random.uniform() < EPSILON:  # greedy
#             # use epsilon-greedy approach to take action
#             actions_value = self.eval_net.forward(x)
#             # print(f"--------- actions_value = {actions_value}  ----- size = {actions_value.shape}")
#             # print(torch.max(actions_value, 1))
#             # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
#             # what we need is the index in this function, representing the action of cart.
#             action = torch.max(actions_value, 1)[1].data.numpy()
#             # print(f"---------- action = {action}")
#
#             action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # return the argmax index
#         else:  # random
#             action = np.random.randint(0, self.actions_num)
#             action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
#         return action
#
#     def store_transition(self, s, a, r, s_):
#         # This function acts as experience replay buffer
#         transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors
#         # print(f" --------- transition = {transition}")
#         # if the capacity is full, then use index to replace the old memory with new one
#         index = self.memory_counter % MEMORY_CAPACITY
#         self.memory[index, :] = transition
#         self.memory_counter += 1
#
#     def learn(self):
#         # Define how the whole DQN works including sampling batch of experiences,
#         # when and how to update parameters of target network, and how to implement
#         # backward propagation.
#
#         # update the target network every fixed steps
#         if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
#             # Assign the parameters of eval_net to target_net
#             self.target_net.load_state_dict(self.eval_net.state_dict()) # model.state_dict() ： 模型的参数
#         self.learn_step_counter += 1
#
#         # Determine the index of Sampled batch from buffer
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # randomly select some data from buffer
#         # extract experiences of batch size from buffer.
#         b_memory = self.memory[sample_index, :]
#         # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
#         # that are convenient to back propagation
#         b_s = Variable(torch.FloatTensor(b_memory[:, :self.states_num])) # batch 中所有的 s_t 保存一起
#         # convert long int type to tensor
#         b_a = Variable(torch.LongTensor(b_memory[:, self.states_num:self.states_num + 1].astype(int)))  # batch中所有的 a_t 保存一起
#         b_r = Variable(torch.FloatTensor(b_memory[:, self.states_num + 1:self.states_num + 2]))  # batch中所有的 r_t 保存一起
#         b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.states_num:]))  # batch中所有的 s_t+1 保存一起
#
#         # calculate the Q value of state-action pair
#         q_eval = self.eval_net(b_s)
#         # print(f" ------ q_eval = {q_eval}")
#         q_eval = q_eval.gather(1, b_a)  # (batch_size, 1)
#       # print(f" --------- q_eval = {q_eval}")
#         # print(q_eval)
#         # calculate the q value of next state
#         q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
#         # select the maximum q value
#         # print(q_next)
#         # q_next.max(1) returns the max value along the axis=1 and its corresponding index
#         q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
#
#         loss = self.loss_func(q_eval, q_target)
#
#         self.optimizer.zero_grad()  # reset the gradient to zero
#         loss.backward()
#         self.optimizer.step()  # execute back propagation for one step
#
#
