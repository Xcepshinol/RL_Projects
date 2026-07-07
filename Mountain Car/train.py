import gymnasium as gym
from stable_baselines3 import PPO
import Policy_network as policy

env = gym.make("MountainCar-v0")

model = policy.QNetwork()
print("Starting training...")
model.train_dqn(env)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=20000)

model.save_model("mountaincar_dqn_model")
env.close()

print("Training complete!")