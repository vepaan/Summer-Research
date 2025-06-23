import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from src.models.networks import MLP, CNN
from torch.distributions import Categorical

class PPOAgent:

    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"PPO Agent is using device: {self.device}")

        agent_cfg = config['agent']
        self.gamma = agent_cfg['gamma_ppo']
        self.clip_epsilon = agent_cfg['clip_epsilon']
        self.ppo_epochs = agent_cfg['ppo_epochs']
        self.entropy_coeff = agent_cfg['entropy_coeff']
        self.gae_lambda = agent_cfg['gae_lambda']

        self.model_type = config['agent']['model_type']
        if self.model_type == 'MLP':
            hidden_size = config['agent']['mlp']['hidden_size']
            self.actor = MLP(state_size, hidden_size, action_size, self.device)
            self.critic = MLP(state_size, hidden_size, 1, self.device)
        elif self.model_type == 'CNN':
            input_shape = config['agent']['cnn']['input_shape']
            conv_channels = config['agent']['cnn']['conv_channels']
            hidden_size = config['agent']['cnn']['hidden_size']
            self.actor = CNN(input_shape, conv_channels, hidden_size, action_size, self.device)
            self.critic = CNN(input_shape, conv_channels, hidden_size, 1, self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=agent_cfg['learning_rate_ppo']
        )
        
        self.memory = []

    
    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        #sample best action from policy
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(state_tensor)
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)

            if evaluation_mode:
                action = dist.mode #take most likely action in eval
                return action.item()
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.critic(state_tensor)
                return action.item(), log_prob, value
        
    
    def store_outcome(self, state, action, log_prob, value, reward, done):
        self.memory.append((state, action, log_prob, value, reward, done))


    def learn(self):
        if len(self.memory) == 0:
            return
        
        states, actions, old_log_probs, values, rewards, dones = zip(*self.memory)
        
        # Convert lists to tensors
        values = torch.cat(values).squeeze(-1).detach()
        old_log_probs = torch.cat(old_log_probs).detach()
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        # GAE Calculation
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            if t == len(rewards) - 1:
                next_value = 0 # Bootstrap with 0 if it's the last state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        
        # Only normalize advantages if the trajectory has more than one step to avoid division by zero.
        if len(self.memory) > 1:
            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pass

        # PPO Update loop
        for _ in range(self.ppo_epochs):

            logits = self.actor(states)
            dist = Categorical(F.softmax(logits, dim=-1))
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            value_preds = self.critic(states).squeeze(-1)

            # Calculate the Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped Surrogate Objective (Policy Loss)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Function Loss (Critic Loss)
            value_loss = F.mse_loss(returns, value_preds)
            
            # Total Loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

            # Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()

        # Crucial for on-policy: Clear the memory after learning
        self.memory.clear()


    def update_action_net(self):
        pass


    def save(self, file_name: str, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save({
            'policy_net': self.actor.state_dict(),
            'value_net': self.critic.state_dict()
        }, os.path.join(folder_path, file_name))
        print(f"\nAgent saved at path: {folder_path} as {file_name}")


    def load(self, file_path: str):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['policy_net'])
        self.critic.load_state_dict(checkpoint['value_net'])
        self.actor.to(self.device).eval()
        self.critic.to(self.device).eval()