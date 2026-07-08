import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_actions=6, memory_size=50_000):
        super().__init__()

        self.memory = deque(maxlen=memory_size)

        self.actions = [
            np.array([-0.8, 0.6, 0.0], dtype=np.float32),  # hard left + gas
            np.array([-0.4, 0.8, 0.0], dtype=np.float32),  # slight left + more gas
            np.array([ 0.0, 1.0, 0.0], dtype=np.float32),  # full gas
            np.array([ 0.4, 0.8, 0.0], dtype=np.float32),  # slight right + more gas
            np.array([ 0.8, 0.6, 0.0], dtype=np.float32),  # hard right + gas
            np.array([ 0.0, 0.0, 0.8], dtype=np.float32),  # brake
        ]

        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
    def preprocess(self, obs):
        x = torch.from_numpy(np.array(obs)).float()

        if x.ndim == 3:
            # Single image
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.ndim == 4:
            # Batch of images
            x = x.permute(0, 3, 1, 2)

        return x / 255.0
    
    def predict_index(self, obs):
        x = self.preprocess(obs)

        with torch.no_grad():
            q_values = self.forward(x)

        return q_values.argmax(dim=1).item()
    
    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            action_idx = random.randrange(len(self.actions))
        else:
            x = self.preprocess(obs)

            with torch.no_grad():
                q_values = self.forward(x)

            action_idx = q_values.argmax(dim=1).item()

        real_action = self.actions[action_idx]

        return action_idx, real_action
    
    def predict(self, obs):
        action_idx = self.predict_index(obs)
        return self.actions[action_idx]
    
    def remember(self, obs, action_idx, reward, next_obs, done):
        self.memory.append((obs, action_idx, reward, next_obs, done))


    def sample_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        obs, actions, rewards, next_obs, dones = zip(*batch)

        return obs, actions, rewards, next_obs, dones

    def train_step(self, target_model, optimizer, criterion, batch_size, gamma=0.99):
        if len(self.memory) < batch_size:
            return

        obs, actions, rewards, next_obs, dones = self.sample_memory(batch_size)

        states = self.preprocess(obs)
        next_states = self.preprocess(next_obs)

        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = target_model(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + gamma * max_next_q * (1 - dones)

        loss = criterion(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def run_episode(self, env, target_model, optimizer, criterion, batch_size, epsilon, gamma=0.99):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_idx, real_action = self.select_action(obs, epsilon)

            next_obs, reward, terminated, truncated, info = env.step(real_action)
            done = terminated or truncated

            self.remember(obs, action_idx, reward, next_obs, done)

            loss = self.train_step(target_model, optimizer, criterion, batch_size, gamma)

            obs = next_obs
            total_reward += reward

        return total_reward
    
    def train(self, env, target_model, optimizer, criterion, batch_size, num_episodes, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, gamma=0.99):
        epsilon = epsilon_start
        for episode in range(num_episodes):
            if episode % 10 == 0:
                target_model.load_state_dict(self.state_dict())
            total_reward = self.run_episode(env, target_model, optimizer, criterion, batch_size, epsilon, gamma)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()



        

    
