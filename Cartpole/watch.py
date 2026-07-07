import gymnasium as gym
from stable_baselines3 import PPO
from policy_network import PolicyNetwork

env = gym.make("CartPole-v1", render_mode="human")

model = PolicyNetwork()
model.load("cartpole_model")

obs, info = env.reset()

while True:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()