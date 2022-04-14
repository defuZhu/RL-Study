from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        hid1_size = act_dim * 10

        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        out = F.softmax(self.fc2(h1))
        return out
