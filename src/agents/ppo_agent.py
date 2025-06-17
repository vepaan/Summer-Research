import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os

from src.models.networks import MLP, CNN
from src.utils.replay_buffer import ReplayBuffer, Experience
from torch.distributions import Categorical

class PPOAgent:

    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"PPO Agent is using device: {self.device}")

        self.model_type = config['agent']['model_type']

        self.model_type = config['agent']['model_type']
        if self.model_type == 'MLP':
            hidden_size = config['agent']['mlp']['hidden_size']
            self.policy_net = MLP(state_size, hidden_size, action_size, self.device)
            self.value_net = MLP(state_size, hidden_size, 1, self.device)
        elif self.model_type == 'CNN':
            input_shape = config['agent']['cnn']['input_shape']
            conv_channels = config['agent']['cnn']['conv_channels']
            hidden_size = config['agent']['cnn']['hidden_size']
            self.policy_net = CNN(input_shape, conv_channels, hidden_size, action_size, self.device)
            self.value_net = CNN(input_shape, conv_channels, hidden_size, 1, self.device)
        else:
            raise ValueError("Unknown model type: ", self.model_type)
        
        # Load PPO-specific hyperparameters from YAML
        agent_cfg = config['agent']
        self.gamma = agent_cfg['gamma_ppo']
        self.clip_epsilon = agent_cfg['clip_epsilon']
        self.ppo_epochs = agent_cfg['ppo_epochs']
        self.entropy_coeff = agent_cfg['entropy_coeff']

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr = config['agent']['learning_rate_ppo']
        )

        self.memory = ReplayBuffer(config['memory']['buffer_size'])
        self.trajectory_log_probs = []
        self.trajectory_values = []

    
    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        #sample best action from policy
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits = self.policy_net(state_tensor)
        probs = F.softmax(logits, dim=1)
        dist = Categorical(probs)

        if evaluation_mode:
            action = probs.argmax(dim=1)
            return action.item()
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value_net(state_tensor)

            #store trajectory info for policy update
            self.trajectory_log_probs.append(log_prob)
            self.trajectory_values.append(value)
            return action.item()
        
    
    def store_outcome(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)


    def learn(self):
        if len(self.memory) == 0:
            return
        
        experiences = self.memory.memory
        batch = Experience(*zip(experiences))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        values = torch.cat(self.trajectory_values).squeeze(1).to(self.device).detach()
        next_values = self.value_net(next_states).squeeze(1).detach()

        gae_lambda = self.config['agent']['gae_lambda']
        gamma = self.gamma

        #deltas (TD errors)
        deltas = rewards + gamma * next_values * (1 - dones) - values

        #gae computation
        advantages = torch.zeros_like(deltas).to(self.device)
        gae = 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        #advantages
        old_log_probs = torch.stack(self.trajectory_log_probs).detach().unsqueeze(1)

        #main PPO update
        for _ in range(self.ppo_epochs):
            logits = self.policy_net(states)
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            
            #clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            #value function loss (critic loss)
            value_preds = self.value_net(states)
            value_loss = F.mse_loss(value_preds, returns)

            #total loss = actor + critic - entropy
            loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.memory.clear()
        self.trajectory_log_probs.clear()
        self.trajectory_values.clear()


    def update_action_net(self):
        pass


    def save(self, file_name: str, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict()
        }, os.path.join(folder_path, file_name))
        print(f"\nAgent saved at path: {folder_path} as {file_name}")


    def load(self, file_path: str):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])