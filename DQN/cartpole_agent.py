import numpy as np
import torch
import torchvision.transforms.functional
from torch.autograd import Variable


class CartpoleAgent:
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        assert isinstance(act_dim, int)
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200  # update target model parameters

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

        self.alg = algorithm

    def sample(self, obs):
        """
        sample an action "for exploration" when given an observation
        :param obs: np.float32
        :return:
        """
        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        """
        predict an action when given an observation

        :param obs:
        :return:
        """
        obs = torch.tensor(obs)
        pred_q = self.alg.predict(obs)
        act = pred_q.argmax().numpy()
        # print(act.shape)
        # print(act)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """

        :param obs: np.float32:  shape of (batch_size, obs_dim)
        :param act: np.int32: shape of (batch_size)
        :param reward: np.float32: shape of (batch_size)
        :param next_obs: np.float32: shape of (batch_size, obs_dim)
        :param terminal: np.float32: shape of (batch_size)
        :return:  loss(float)
        """
        """
        prepare data for learn
        """

        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss
