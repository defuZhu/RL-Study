import copy

import torch.optim
from torch.autograd import Variable


class DDPG(object):
    def __init__(self, model, gamma=None, tau=None, actor_lr=None, crtic_lr=None):
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = crtic_lr

        self.model = model
        self.target_model = copy.deepcopy(model)

        self.actor_optimizer = torch.optim.Adam(self.model.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_model.parameters(), lr=crtic_lr)

        self.loss_fn = torch.nn.MSELoss()

    def predict(self, obs):
        """
         use actor_model predict action
        :param obs:
        :return:
        """
        obs = torch.tensor(obs)
        return self.model.policy(obs)


    def learn(self, obs, action, reward, next_obs, terminal):
        """
        use DDPG update actor and critic
        :param obs:
        :param action:
        :param reward:
        :param next_obs:
        :param terminal:
        :return:
        """
        obs = Variable(torch.FloatTensor(obs))
        action = Variable(torch.FloatTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        next_obs = Variable(torch.FloatTensor(next_obs))
        terminal = Variable(torch.LongTensor(terminal))

        actor_loss = self._actor_learn(obs)
        critic_loss = self._critic_learn(obs, action, reward, next_obs, terminal)
        return actor_loss, critic_loss

    def _actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        loss = torch.mean(-1.0 * Q)
        # print(f"loss = {loss}")
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss.item()


    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        reward = torch.reshape(reward, [32, 1])
        terminal = torch.reshape(terminal, [32, 1])

        with torch.no_grad():
            next_action = self.target_model.policy(next_obs)
            next_Q = self.target_model.value(next_obs, next_action)
            target_Q = reward + (1.0 - terminal) * self.gamma * next_Q


        Q = self.model.value(obs, action)
        # print(Q)
        # print(target_Q)
        loss = self.loss_fn(Q, target_Q)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return loss.item()


    def sync_target(self, decay=None):
        if decay == None:
            decay = 1.0 - self.tau

        # hard update
        self.target_model.actor_model.load_state_dict(self.model.actor_model.state_dict())
        self.target_model.critic_model.load_state_dict(self.model.critic_model.state_dict())

        # target_actor_model
        # target_actor_vars = dict(self.target_model.actor_model.named_parameters())
        for name, var in self.model.actor_model.named_parameters():
            self.target_model.actor_model.state_dict()[name].data = self.target_model.actor_model.state_dict()[name].data * (1.0 - self.tau) + \
                                                                  var.data * self.tau

        # # target_critic_model
        # target_critic_vars = dict(self.target_model.critic_model.named_parameters())
        for name, var in self.model.critic_model.named_parameters():
            self.target_model.critic_model.state_dict()[name].data = self.target_model.critic_model.state_dict()[name].data * (1.0 - self.tau) + \
                                                                  var.data * self.tau
        # for name, param in self.target_model.actor_model.named_parameters():
        #     print(self.target_model.actor_model.state_dict)
