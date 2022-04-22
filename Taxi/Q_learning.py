"""
    Reference:
            https://github.com/e10101/deep-reinforcement-learning-resources

"""


import numpy as np
import time
import PIL
from PIL import ImageDraw, ImageFont
import matplotlib
import gym
import imageio
from absl import logging
from tqdm import tqdm
from IPython.display import clear_output

from gym.envs.toy_text.taxi import MAP as taxi_map

action_names = [
    'South (↓)',
    'North (↑)',
    'East (→)',
    'West (←)',
    'Pickup',
    'Drop off',
]

learning_rate = 0.1
gamma = 0.99
epsilon = 0.1
episodes = int(1e5)
# env = gym.make("Taxi-v3")


# evaluate an episode
def play_env(env, policy=lambda s, env=gym.make("Taxi-v3"): env.action_space.sample(), sleep_time=0.1, env_seed=None):
    if env_seed is not None:
        env.seed(env_seed)

    states = []
    rewards = []
    actions = []
    state = env.reset()
    states.append(state)
    max_steps = env.spec.max_episode_steps
    total_reward = 0
    is_done = False
    current_step = 0

    while not is_done:
        action = policy(state)
        state, reward, is_done, info = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)

        total_reward += reward
        current_step += 1
        # clear_output(wait=True)

        print("Step: {:03d}/{}, Reward: {}\n".format(current_step, max_steps, total_reward))

        env.render()

        print("\n{}".format(list(env.decode(state))))
        time.sleep(sleep_time)

    if current_step < max_steps:
        print('\nResult: Done with {} steps and total reward is {}.'.format(
            current_step,
            total_reward,
        ))
    else:
        print('\nResult: Unsolved')

    return states, rewards, actions


def get_char_txt(char_row, char_col, char='█'):
    txt = ''
    for r in range(char_row + 2):
        for c in range(char_col * 2 + 2):
            if (char_row + 1) == r and (char_col * 2 + 1) == c:
                txt += char
            else:
                txt += ' '
        txt += '\n'
    # print(txt)
    return txt


def get_char_by_index(idx: int):
    if idx == 0:
        return 'R'
    elif idx == 1:
        return 'G'
    elif idx == 2:
        return 'Y'
    elif idx == 3:
        return 'B'
    elif idx == 4:
        return '_'
    return ' '


def get_char_pos_by_index(idx: int, taxi_row: int, taxi_col: int):
    if idx == 0:
        return [0, 0]
    elif idx == 1:
        return [0, 4]
    elif idx == 2:
        return [4, 0]
    elif idx == 3:
        return [4, 3]

    return [taxi_row, taxi_col]


def get_char_color_by_index(idx: int):
    if idx == 0:
        return (255, 0, 0)  # Red
    elif idx == 1:
        return (0, 255, 0)  # Green
    elif idx == 2:
        return (255, 255, 0)  # Yellow
    elif idx == 3:
        return (0, 0, 255)  # Blue


def enhance_frame(frame: np.ndarray, env, main_text=None, state=None, side_text=None, done=False):
    if main_text is None:
        return frame

    image = PIL.Image.fromarray(frame).convert("RGB")

    draw = ImageDraw.Draw(image, 'RGB')

    font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    font = ImageFont.truetype(font_file, 24)

    draw_offset = (30, 30)
    side_offset = (220, 30)
    side_font_size = 20
    side_font = ImageFont.truetype(font_file, side_font_size)

    taxi_color = (255, 255, 0)
    passenger_color = (0, 0, 255)
    dest_color = (255, 0, 255)
    taxi_with_passenger_color = (0, 255, 0)

    if state is not None:
        # print(f"state : {state}")
        [taxi_row, taxi_col, passenger_location, destination] = list(env.decode(state))
        print([taxi_row, taxi_col, passenger_location, destination])
        taxi_txt = get_char_txt(taxi_row, taxi_col)

        taxi_color = taxi_with_passenger_color if (passenger_location == 4 and not done) else taxi_color
        draw.text(draw_offset, taxi_txt, font=font, fill=taxi_color, stroke_width=1, stroke_fill=(100, 100, 100))

        draw.text(draw_offset, main_text, font=font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))

        # Draw passenger
        passenger_char = get_char_by_index(passenger_location)
        [passenger_row, passenger_col] = get_char_pos_by_index(passenger_location, taxi_row, taxi_col)
        passenger_txt = get_char_txt(passenger_row, passenger_col, char=passenger_char)
        # passenger_color = get_char_color_by_index(passenger_location)
        # print('passenger_txt', passenger_txt)
        draw.text(draw_offset, passenger_txt, font=font, fill=passenger_color, stroke_width=1,
                  stroke_fill=(255, 255, 255))

        # Draw destination
        dest_char = get_char_by_index(destination)
        [dest_row, dest_col] = get_char_pos_by_index(destination, taxi_row, taxi_col)
        dest_txt = get_char_txt(dest_row, dest_col, char=dest_char)
        # dest_color = get_char_color_by_index(destination)
        # print('dest_txt', dest_txt)
        draw.text(draw_offset, dest_txt, font=font, fill=dest_color, stroke_width=1, stroke_fill=(255, 255, 255))

    else:
        # Draw background
        draw.text(draw_offset, main_text, font=font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))

    if side_text is not None:
        draw.text(side_offset, side_text, font=side_font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))

    return np.array(image)


def get_timestamp():
    now = int(round(time.time() * 1000))
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
    return time_stamp


def create_states_video(env, states, rewards, actions, filename=None, fps=30,
                        env_name='Taxi-v3', freeze_seconds=0, freeze_begin_seconds=0, step=None):

    if filename is None:
        filename = str(get_timestamp())

    filename = filename + '.mp4'
    logging.info('Env: %s', env_name)
    logging.info('Filename: %s', filename)
    map_txt = '\n'.join(taxi_map)

    with imageio.get_writer(filename, fps=fps) as video:
        logging.info('Begin')
        total_reward = 0.0
        frame_idx = 0

        for idx, (state, reward, action) in enumerate(zip(states, rewards, actions)):
            done = reward == 20
            # Freeze frame for a few seconds - At beginning
            if idx == 0 and freeze_begin_seconds > 0:
                text = f'Env: {env_name}'
                if step is not None:
                    text += f'\nStp: {step}'
                text += f'\nFrm: {frame_idx}'
                text += f'\nRw:  {total_reward:.2f}'

                frame = np.full((270, 480), 240.0)
                frame = enhance_frame(frame, env, '{}'.format(map_txt), side_text=text, state=state)

                for _ in range(fps * freeze_begin_seconds):
                    video.append_data(frame)

            action_name = action_names[action]

            total_reward += reward

            text = f'Env: {env_name}'
            if step is not None:
                text += f'\nStp: {step}'
            text += f'\nFrm: {frame_idx}'
            text += f'\nRw:  {total_reward:.2f}'
            text += f'\nAct: {action_name}'

            if done:
                text += f'\n\nDone!\nFeb 13, 2022'

            frame = np.full((270, 480), 240.0)
            frame = enhance_frame(frame, env, '{}'.format(map_txt), side_text=text, state=state, done=done)

            video.append_data(frame)

            frame_idx += 1

            # Freeze frame for a few seconds
            if frame_idx + 1 >= len(states) and freeze_seconds > 0:
                for _ in range(fps * freeze_seconds):
                    video.append_data(frame)

    logging.info('All done')
    return filename


def run_episode(env, q_table):
    state = env.reset()

    done = False
    rng = np.random.default_rng()
    while not done:
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)

        current_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_table[state, action] = current_value + learning_rate * (next_max * gamma + reward - current_value)

        state = next_state

    return q_table


def main():
    logging.set_verbosity(logging.INFO)
    env = gym.make("Taxi-v3")

    # Q-learning
    q_table_shape = [env.observation_space.n, env.action_space.n]

    q_table = np.zeros(q_table_shape)

    # train
    for i in tqdm(range(episodes)):
        q_table = run_episode(env, q_table)

    # eval
    states, rewards, actions = play_env(env,
                                        policy=lambda s: np.argmax(q_table[s]),
                                        sleep_time=0.5,
                                        env_seed=1,
                                        )

    create_states_video(env, states, rewards, actions, filename='taxi', fps=2, env_name='Taxi-v3', freeze_seconds=3,
                        freeze_begin_seconds=2)


if __name__ == '__main__':
    main()
    # env = gym.make("Taxi-v3")
    # play_env(env)
