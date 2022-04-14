import torch
import torch.nn.functional as F
from torch import nn


class PolicyGradient(object):
    def __init__(self, model, lr):
        self.model = model
        # self.gamma = gamma
        self.lr = lr

        # define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss()

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, action, reward):

        # a trajectory obs: s1, s2, s3, ...
        """
        output: s1 --> p(a1|s1), p(a2|s1), ...
                s2 --> p(a1|s2), p(a2|s2), ...
                ...
        """
        act_prob = self.model(obs)

        # print(act_prob)
        log_prob = 0
        for i in range(len(act_prob)):
            log_prob += (-1.0) * torch.log(act_prob[i][action[i]]) * reward[i]

        # log_prob = torch.sum(-1.0*torch.log(act_prob) * torch.nn.functional.one_hot(action) * reward)

        # log_prob = self.loss_func(act_prob, action)


        # print(log_prob)
        # log_probility = nn.CrossEntropyLoss(act_probility, action)
        # cost = log_prob * reward
        # print(f"cost = {cost}")
        # loss = torch.sum(log_prob)

        self.optimizer.zero_grad()
        log_prob.backward()
        self.optimizer.step()

        return log_prob.item()
