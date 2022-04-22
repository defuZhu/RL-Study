from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        hide_size = 128
        self.fc1 = nn.Linear(obs_dim, hide_size)
        self.fc2 = nn.Linear(hide_size, hide_size)
        self.fc3 = nn.Linear(hide_size, act_dim)


    def forward(self, x):
        # print(f"x : {x}")
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h1))
        output = self.fc3(h1)
        return output
