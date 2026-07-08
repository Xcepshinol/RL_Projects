import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
import torch.optim as optim
import cnn
import torch.nn as nn

env = gym.make("CarRacing-v3")

model = cnn.CNN()
target_model = cnn.CNN()
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

model.train(
    env=env,
    target_model=target_model,
    optimizer=optimizer,
    criterion=criterion,
    batch_size=32,
    num_episodes=500
)

model.save("carracing_dqn.pth")
env.close()