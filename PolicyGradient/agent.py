import numpy as np
import torch
from torch.autograd import Variable


class Agent(object):
    def __init__(self, algorithm, act_dim):
        self.alg = algorithm
        self.act_dim = act_dim

    def sample(self, obs):
        obs = torch.tensor(obs)
        act_prob = self.alg.model(obs).detach()
        # print(act_prob)
        act_prob = torch.squeeze(act_prob, dim=0)
        # print(act_prob)
        act = np.random.choice(range(self.act_dim), p=act_prob.numpy())
        return act

    def predict(self, obs):
        #select probility max action
        obs = torch.tensor(obs)
        act_prob = self.alg.model(obs)
        # print(act_prob)

        act = act_prob.argmax(0).item()
        return act

    def learn(self, obs, act, reward):

        # preprocess
        obs = Variable(torch.FloatTensor(obs))
        act = Variable(torch.LongTensor(act))
        reward = Variable(torch.FloatTensor(reward))


        # update
        cost = self.alg.learn(obs, act, reward)

        return cost