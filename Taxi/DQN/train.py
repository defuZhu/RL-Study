import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Model
from algorithm import DQN
from agent import Agent
from CartPole.replay_memory import ReplayMemory

import time
import PIL
from PIL import ImageDraw, ImageFont
import matplotlib
import imageio
from absl import logging
from IPython.display import clear_output

from gym.envs.toy_text.taxi import MAP as taxi_map
from Taxi.Q_learning import (get_char_txt,
                             get_char_by_index,
                             get_char_color_by_index,
                             get_char_pos_by_index,
                             create_states_video,
                             enhance_frame)

LEARN_FREQ = 5
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.99


global_train_step = 0


def run_train_episode(agent, env, rpm, writer):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        global global_train_step

        obs_list = list(env.decode(obs))  # obs must be float32 type

        # action = env.action_space.sample()

        action = agent.sample(obs_list)
        # print(f"action : {action}")

        next_obs, reward, done, _ = env.step(action)  # action must be int type

        next_obs_list = list(env.decode(next_obs))

        rpm.append(obs_list, action, reward, next_obs_list, done)
        # print(next_obs)
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
            global_train_step += 1
            writer.add_scalar("train_loss", train_loss, global_train_step)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


def test(env, policy=lambda s, env=gym.make("Taxi-v3"): env.action_space.sample(), sleep_time=0.1, env_seed=None):
    if env_seed is not None:
        env.seed(env_seed)

    states = []
    rewards = []
    actions = []
    state = env.reset()
    state_list = list(env.decode(state))

    states.append(state)
    max_steps = env.spec.max_episode_steps
    total_reward = 0
    is_done = False
    current_step = 0

    while not is_done:
        action = policy(state_list)  # policy() input must be [ , , , ]
        state, reward, is_done, info = env.step(action)
        state_list = list(env.decode(state))

        states.append(state)  # states must be an integer, for plot
        rewards.append(reward)
        actions.append(action)

        total_reward += reward
        current_step += 1
        # clear_output(wait=True)

        print("Step: {:03d}/{}, Reward: {}".format(current_step, max_steps, total_reward))
        print("{}".format(list(env.decode(state))))
        env.render()  # 渲染
        print("\n")

        time.sleep(sleep_time)

    if current_step < max_steps:
        print('\nResult: Done with {} steps and total reward is {}.'.format(
            current_step,
            total_reward,
        ))
    else:
        print('\nResult: Unsolved')

    return states, rewards, actions


def main():
    env = gym.make("Taxi-v3")
    obs_dim = 4
    act_dim = env.action_space.n
    print(f"obs_dim = {obs_dim}, act_dim = {act_dim}")

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    model = Model(obs_dim, act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(alg, act_dim, 0.1, 1e-6)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)
    max_episode = int(50000)

    # start
    episode = 0
    writer = SummaryWriter("logs")
    for episode in tqdm(range(max_episode)):
        total_reward = run_train_episode(agent, env, rpm, writer)
        episode += 1

        # save model
        if (episode % 10000 == 0):
            torch.save(model.state_dict(), "model_{}.pth".format(episode))
        # eval_reward = run_evaluate_episode(agent, env, render=True)
        # print(f"episode = {episode}, total_reward = {total_reward}")
    writer.close()


def replay_model():
    env = gym.make("Taxi-v3")
    obs_dim = 4
    act_dim = env.action_space.n

    # 导入模型和参数
    model = Model(obs_dim, act_dim)
    model.load_state_dict(torch.load("model_50000.pth"))

    # eval
    algorithm = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(algorithm, act_dim, 0.1, 1e-6)

    logging.set_verbosity(logging.INFO)

    states, rewards, actions = test(env,
                                    policy=lambda s: agent.predict(s),
                                    sleep_time=0.5,
                                    # env_seed=1,  # if env_seed == 1, there will be only a same start state
                                    )

    create_states_video(env, states, rewards, actions, filename='taxi', fps=2, env_name='Taxi-v3', freeze_seconds=3,
                        freeze_begin_seconds=2)


if __name__ == '__main__':
    # main()   # train model

    # load trained model and test
    replay_model()
