import numpy as np
import random
import copy
from math import sqrt

import gymnasium as gym
from PIL import Image
import cv2


def show_frame_cv2(frame, window_name="Frozen Lake", delay=100):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(delay)


def qtable_directions_map():
    qtable_val_max = Q_table.max(axis=1).reshape(dimension, dimension)
    qtable_best_action = np.argmax(Q_table, axis=1).reshape(dimension, dimension)

    directions = {0: "<", 1: "v", 2: ">", 3: "^"}
    qtable_directions = np.full(qtable_best_action.shape, " ", dtype=str)

    eps = np.finfo(float).eps

    for r in range(dimension):
        for c in range(dimension):
            if qtable_val_max[r][c] > eps:
                qtable_directions[r][c] = directions[qtable_best_action[r][c]]

    return qtable_val_max, qtable_directions


def show_qtable_cv2():
    _, arrows = qtable_directions_map()

    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cell_size = 100

    for r in range(dimension):
        for c in range(dimension):
            x1 = c * cell_size
            y1 = r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(
                img,
                arrows[r][c],
                (x1 + 35, y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

    cv2.imshow("Q-table policy", img)
    cv2.waitKey(1)


def policy_greedy(Q_table, state):
    return np.argmax(Q_table[state])


def policy_epsilon_greedy(Q_table, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    return policy_greedy(Q_table, state)


def sample_trajectory(epsilon, episode_num):
    curr_state, _ = env.reset()

    trajectory = []
    terminated = False
    truncated = False

    render_episode = episode_num % 50 == 0

    while not (terminated or truncated):
        if render_episode:
            frame = env.render()
            show_frame_cv2(frame, "Training Frozen Lake", delay=50)

        action = policy_epsilon_greedy(Q_table, curr_state, epsilon)

        new_state, reward, terminated, truncated, _ = env.step(action)

        trajectory.append((curr_state, action, reward))

        curr_state = new_state

        row = curr_state // dimension
        col = curr_state % dimension
        state_freq_table[row][col] += 1

    if render_episode:
        frame = env.render()
        show_frame_cv2(frame, "Training Frozen Lake", delay=300)
        show_qtable_cv2()

    return trajectory


def calculate_returns(trajectory):
    returns = []

    for index in range(len(trajectory)):
        return_val = 0
        n = 0

        for trajectory_point in trajectory[index:]:
            return_val += (discount_factor ** n) * trajectory_point[2]
            n += 1

        returns.append(return_val)

    return returns


def update_Qtable(Q_table, trajectory, returns):
    for index in range(len(trajectory)):
        state, action, reward = trajectory[index]
        return_val = returns[index]

        Q_table[state][action] = Q_table[state][action] + learning_rate * (
            return_val - Q_table[state][action]
        )


env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="rgb_array"
)

obs, info = env.reset()

state_space = env.observation_space.n
action_space = env.action_space.n

Q_table = np.zeros((state_space, action_space))

dimension = int(sqrt(state_space))
state_freq_table = np.zeros((dimension, dimension))

num_episodes = 1500

epsilon_max = 0.9
epsilon_min = 0.0
epsilon_decay_rate = 0.001

discount_factor = 0.95
learning_rate = 0.05

avg_rewards = []
average_reward = 0

for episode_num in range(num_episodes):
    epsilon = max(epsilon_max - epsilon_decay_rate * episode_num, epsilon_min)

    trajectory = sample_trajectory(epsilon, episode_num)

    returns = calculate_returns(trajectory)

    update_Qtable(Q_table, trajectory, returns)

    average_reward += trajectory[-1][2]

    if episode_num % 50 == 0:
        avg_rewards.append(average_reward / 50)
        average_reward = 0
        print(f"Episode {episode_num}, epsilon={epsilon:.3f}, avg reward={avg_rewards[-1]:.3f}")

cv2.destroyAllWindows()

print("Final Q-table:")
print(Q_table)

print("Average rewards:")
print(avg_rewards)