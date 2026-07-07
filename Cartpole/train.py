import gymnasium as gym
from stable_baselines3 import PPO
import policy_network as policy

env = gym.make("CartPole-v1")

model = policy.PolicyNetwork()
print("Starting training...")
model.run(env, num_episodes=1000)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=20000)

model.save("cartpole_model")
env.close()

print("Training complete!")