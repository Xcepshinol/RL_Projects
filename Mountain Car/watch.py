import gymnasium as gym
from stable_baselines3 import PPO
from Policy_network import QNetwork

env = gym.make("MountainCar-v0", render_mode="human")

model = QNetwork()
model.load()

obs, info = env.reset()

while True:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()