
import gym
import numpy as np

from agent import Agent
from env import ContinuousCartPoleEnv
from algorithm import DDPG
from model import Model
from CartPole.replay_memory import ReplayMemory


ACTOR_LR = 1E-3
CRITIC_LR = 1E-3
GAMMA = 0.99
TAU = 0.9  # soft update
MEMORY_SIZE = int(1E6)
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 32
REWARD_SCALE = 0.1
NOISE = 0.05
TRAIN_EPISODE = 6E3


# train an episode
def run_episode(agent, env, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        action = agent.predict(obs)
        action = np.clip(np.random.normal(action, NOISE), -1.0, 1.0)

        next_obs, reward, done, _ = env.step(action)
        rpm.append(obs, action, REWARD_SCALE*reward, next_obs, done)

        if len(rpm) > MEMORY_WARMUP_SIZE and steps % 5 == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        obs = next_obs
        total_reward += reward

        if done or steps >= 200:
            break
    return total_reward


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            action = agent.predict(obs)
            action = np.clip(action, -1.0, 1.0)

            steps += 1
            obs, reward, done, _ = env.step(action)

            total_reward += reward

            if render:
                env.render()
            if done or steps >= 200:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

def main():
    env = ContinuousCartPoleEnv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # print(act_dim)

    model = Model(obs_dim, act_dim)
    algorithm = DDPG(model, GAMMA, TAU, ACTOR_LR, CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent, env, rpm)

    episode = 0
    while episode < TRAIN_EPISODE:
        for i in range(50):
            total_reward = run_episode(agent, env, rpm)
            episode += 1

        eval_reward = evaluate(env, agent, render=True)
        print(f"episode = {episode}, eval_reward = {eval_reward}")


if __name__ == '__main__':
    main()

