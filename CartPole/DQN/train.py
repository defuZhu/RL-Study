import numpy as np
import gym
from dqn import DQN
# 1. Define some Hyper Parameters
# BATCH_SIZE = 32  # batch size of sampling process from buffer
# LR = 0.01  # learning rate
# EPSILON = 0.9  # epsilon used for epsilon greedy approach
# GAMMA = 0.9  # discount factor
# TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target netowrk updates
# MEMORY_CAPACITY = 2000  # The capacity of experience replay buffer\

from cartpole_agent import CartpoleAgent
from cartpole_model import CartPoleModel
from CartPole.replay_memory import ReplayMemory

LEARN_FREQ = 5
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.99


def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)
        print(next_obs)
        if(len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


def run_evaluate_episode(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)



def main():
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"obs_dim = {obs_dim}, act_dim = {act_dim}")

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    model = CartPoleModel(obs_dim, act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = CartpoleAgent(alg, act_dim, 0.1, 1e-6)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    max_episode = 4000

    # start
    episode = 0
    while episode < max_episode:
        for i in range(50):
            total_reward = run_train_episode(agent, env, rpm)
            episode += 1

        eval_reward = run_evaluate_episode(agent, env, render=True)
        print(f"episode = {episode}, test_reward = {eval_reward}")


if __name__ == '__main__':
    main()

#
# env = gym.make("CartPole-v0")  # Use cartpole game as environment
# env = env.unwrapped
# N_ACTIONS = env.action_space.n  # 2 actions   left or right
# N_STATES = env.observation_space.shape[0]  # 4 states
#
# print(f"actions = {N_ACTIONS}")
#
# print(env.observation_space)
#
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
#                               int) else env.action_space.sample().shape  # to confirm the shape
#
# def run_episode(env, memory_size, dqn):
#     total_reward = 0
#     obs = env.reset()
#     step = 0
#     while True:
#         step += 1
#         action = dqn.choose_action(obs)
#         next_obs, reward, done, _ = env.step(action)
#         dqn.store_transition(obs, action, reward, next_obs)
#
#
#
# '''
# --------------Procedures of DQN Algorithm------------------
# '''
# # create the object of DQN class
# dqn = DQN(N_STATES, N_ACTIONS, ENV_A_SHAPE)
#
# # Start training
# print("\nCollecting experience...")
# for i_episode in range(1000):
#     # play 400 episodes of cartpole game
#
#     s = env.reset()
#     ep_r = 0
#     while True:
#         # 渲染
#         # env.render()
#         # take action based on the current state
#         a = dqn.choose_action(s)
#         # obtain the reward and next state and some other information
#         s_, r, done, info = env.step(a)
#
#         # modify the reward based on the environment state
#         # x, x_dot, theta, theta_dot = s_
#         # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         # r = r1 + r2
#
#         # store the transitions of states
#         dqn.store_transition(s, a, r, s_)
#
#         ep_r += r
#         # if the experience replay buffer is filled, DQN begins to learn or update
#         # its parameters.
#         if dqn.memory_counter > MEMORY_CAPACITY:
#             dqn.learn()
#             # if done:
#                 # print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
#
#         if done:
#             # if game is over, then skip the while loop.
#             break
#         # use next state to update the current state.
#         s = s_
#
#     test_reward = 0
#     if i_episode % 50 == 0:
#         for _ in range(5):
#             obs = env.reset()
#             while True:
#                 action = dqn.predict(obs)
#                 next_s, reward, done, _ = env.step(action)
#                 test_reward += reward
#                 env.render()
#                 if done:
#                     break
#         print(f" ------ episode = {i_episode},  test_reward = {test_reward} ")
#
