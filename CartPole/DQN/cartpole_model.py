import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CartPoleModel(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super(CartPoleModel, self).__init__()
        hid1_size = 128
        hid2_size = 128

        # three fully connection network
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        return Q

# class Net(nn.Module):
#     def __init__(self, N_STATES, N_ACTIONS):
#         # Define the network structure, a very simple fully connected network
#         super(Net, self).__init__()
#         # Define the structure of fully connected network
#         self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
#         self.fc1.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc1
#         self.out = nn.Linear(10, N_ACTIONS)  # layer 2
#         self.out.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc2
#
#     def forward(self, x):
#         # Define how the input data pass inside the network
#         x = self.fc1(x)
#         x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value


# class Q_Model(nn.Module):
#     def __init__(self, states, actions):
#         super(Q_Model, self).__init__()
#         # self.linear1 = nn.Linear(states, 10)
#         # self.linear1.weight.data.normal_(0, 0.1)
#         #
#         # self.linear2 = nn.Linear(10, actions)
#         # self.linear2.weight.data.normal_(0, 0.1)
#         self.seq = nn.Sequential(
#             nn.Linear(states, 10),
#             nn.Linear(states,10).weight.data.normal_(0, 0.1),
#             nn.ReLU(),
#             nn.Linear(10, actions),
#             nn.Linear(10, actions).weight.data.normal_(0, 0.1)
#         )
#
#     def forward(self, x):
#         x = self.seq(x)
#         return x
#
#
# if __name__ == '__main__':
#     states, actions = 4, 2
#     model = Q_Model(states, actions)
#     input = torch.ones(states,1)
#     print(input)
#     output = model(input)
#     print(output)
