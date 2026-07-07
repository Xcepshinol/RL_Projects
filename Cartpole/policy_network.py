from fileinput import filename

import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)  # logits

    def selectAction(self, obs):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def runEpisode(self, env):
        obs, info = env.reset()

        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_prob = self.selectAction(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            done = terminated or truncated

        return log_probs, rewards

    def computeReturns(self, rewards, gamma=0.99):
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # helps stabilize training
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def updatePolicy(self, optimizer, log_probs, returns):
        loss = 0

        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def run(self, env, num_episodes=1000):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        for episode in range(num_episodes):
            log_probs, rewards = self.runEpisode(env)

            returns = self.computeReturns(rewards)

            self.updatePolicy(optimizer, log_probs, returns)

            print(f"Episode {episode}, reward: {sum(rewards)}")
    
    def save(self, filename="cartpole_model.pth"):
        torch.save(self.state_dict(), filename)

    def load(self, filename="cartpole_model.pth"):
        self.load_state_dict(torch.load(filename))
        self.eval()
    
    def predict(self, obs):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item()

# env = gym.make("CartPole-v1")

# policy = PolicyNetwork()
# policy.run(env, num_episodes=1000)
# env.close()