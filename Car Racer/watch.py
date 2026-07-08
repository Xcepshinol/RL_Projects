import gymnasium as gym
from stable_baselines3 import PPO
import cnn

env = gym.make("CarRacing-v3", render_mode="human")

model = cnn.CNN()
model.load("car_racing_model")

obs, info = env.reset()

while True:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()