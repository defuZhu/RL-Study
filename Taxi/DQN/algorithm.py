import copy

import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class DQN(object):
    def __init__(self, model, gamma, lr):
        self.model = model
        self.target_model = copy.deepcopy(model)

        # on gpu
        self.model, self.target_model = self.model.to(device), self.target_model.to(device)

        self.gamma = gamma
        self.lr = lr

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        #on gpu
        self.loss_fn = self.loss_fn.to(device)


    def predict(self, obs):
        obs = obs.to(device)
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        obs = Variable(torch.FloatTensor(obs))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        next_obs = Variable(torch.FloatTensor(next_obs))
        terminal = Variable(torch.LongTensor(terminal))

        # add a dimension
        action = torch.unsqueeze(action, 1)
        terminal = torch.unsqueeze(terminal, 1)
        reward = torch.unsqueeze(reward, 1)

        obs = obs.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_obs = next_obs.to(device)
        terminal = terminal.to(device)

        pred_value = self.model(obs).gather(1, action)


        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)[0]
            target = reward + (1 - terminal) * self.gamma * max_v

        loss = self.loss_fn(pred_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
