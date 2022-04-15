import numpy as np
import torch


class Agent(object):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.alg = algorithm
        self.alg.sync_target(decay=0)
        self.sync_target_step = 10
        self.learn_step = 0

    def predict(self, obs):
        # print(obs)
        obs = torch.tensor(obs)
        act = self.alg.predict(obs)
        return act.item()

    def learn(self, obs, act, reward, next_obs, terminal):
        cost = self.alg.learn(obs, act, reward, next_obs, terminal)
        self.learn_step += 1
        if self.learn_step % self.sync_target_step == 0:
            self.alg.sync_target()

        return cost