import gym
import numpy as np

from agent import Agent
from algorithm import PolicyGradient
from model import Model

LEARNING_RATE = 1E-3

TRAIN_EPISODE_TOTAL = 1000

# train an episode
def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        reward_list.append(reward)

        if done:
            # print(reward)
            break
    return obs_list, action_list, reward_list


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        epsiode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            epsiode_reward += reward

            if render:
                env.render()

            if done:
                break

        eval_reward.append(epsiode_reward)
    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=1.0):
    # G_i = r_i + gamma * G_i+1
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += gamma * reward_list[i + 1]
    return np.array(reward_list)


def main():
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = Model(obs_dim, act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, act_dim)

    for i in range(TRAIN_EPISODE_TOTAL):
        obs_list, action_list, reward_list = run_episode(env, agent)

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if(i+1)%100 == 0:
            total_reward = evaluate(env, agent, True)
            print(f"train_episode = {i}, total_reward = {total_reward}")


if __name__ == '__main__':
    main()