"""

    two models:
        critic model - Q network
        actor model  - policy network

"""
import torch
from torch import nn
import torch.nn.functional as F




class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        hid_size = 100
        self.fc1 = nn.Linear(obs_dim + act_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, 1)  # output feature = 1, given s and a

    def forward(self, obs, act):
        # print(f"obs = {obs}")
        # print(f"act = {act}")
        input = torch.concat([obs, act], dim=1)
        h = F.relu(self.fc1(input))
        Q = self.fc2(h)
        return Q


class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        hid_size = 100

        self.fc1 = nn.Linear(obs_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, act_dim)

    def forward(self, obs):
        # print(obs)
        h = F.relu(self.fc1(obs))
        Q = torch.tanh(self.fc2(h))
        return Q


class Model(object):
    def __init__(self, obs_dim, act_dim):
        self.actor_model = ActorModel(obs_dim, act_dim)
        self.critic_model = CriticModel(obs_dim, act_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()
