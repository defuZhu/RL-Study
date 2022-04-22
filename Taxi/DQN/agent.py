import numpy as np
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        self.alg = algorithm
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200  # update target model parameters

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement


    def sample(self, obs):
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
        obs = torch.tensor(obs, dtype=torch.float)
        pred_q = self.alg.predict(obs)
        act = pred_q.argmax().numpy()
        # print(act.shape)
        # print(act)
        return int(act)

    def learn(self, obs, action, reward, next_obs, terminal):

        if(self.global_step % self.update_target_steps == 0):
            self.alg.sync_target()

        loss = self.alg.learn(obs, action, reward, next_obs, terminal)
        self.global_step += 1
        return loss