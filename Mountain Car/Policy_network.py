import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


    def select_action(self, q_net, obs, env, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = q_net(obs_tensor)
            return torch.argmax(q_values).item()


    def train_step(self,q_net, optimizer, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return None

        batch = random.sample(memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        current_q = q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = q_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    def train_dqn(self,
        env,
        episodes=500,
        gamma=0.99,
        lr=1e-3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=50_000
    ):

        q_net = QNetwork()
        optimizer = optim.Adam(q_net.parameters(), lr=lr)
        memory = deque(maxlen=memory_size)

        for episode in range(episodes):
            obs, info = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(q_net, obs, env, epsilon)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                memory.append((obs, action, reward, next_obs, done))

                obs = next_obs
                total_reward += reward

                self.train_step(q_net, optimizer, memory, batch_size, gamma)

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print(f"Episode {episode}, reward: {total_reward}, epsilon: {epsilon:.3f}")

        env.close()

        return q_net


    def save_model(q_net, filename="mountaincar_dqn.pth"):
        torch.save(q_net.state_dict(), filename)


    def load(self, filename="mountaincar_dqn.pth"):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def predict(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(obs_tensor)
            action = torch.argmax(q_values).item()
        return action