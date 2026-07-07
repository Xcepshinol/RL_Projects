import torch
import torch.nn as nn
import torch.optim as optim
import os

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
       
    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = torch.unsqueeze(dones, 0)

        current_q_values = self.model(states)
        next_q_values = self.model(next_states)

        targets = current_q_values.clone()

        for idx in range(len(dones)):
            action_idx = torch.argmax(actions[idx]).item()

            if not dones[idx]:
                targets[idx][action_idx] = rewards[idx] + self.gamma * torch.max(next_q_values[idx])
            else:
                targets[idx][action_idx] = rewards[idx]

        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, targets.detach())
        loss.backward()
        self.optimizer.step()