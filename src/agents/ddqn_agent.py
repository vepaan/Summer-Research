import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os

from src.models.networks import MLP, CNN
from src.utils.replay_buffer import ReplayBuffer, Experience

class DDQNAgent:

    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"DDQN Agent is using device: {self.device}")

        self.model_type = config['agent']['model_type']

        if self.model_type == 'MLP':
            hidden_size = config['agent']['mlp']['hidden_size']
            self.policy_net = MLP(state_size, hidden_size, action_size, self.device)
            self.action_net = MLP(state_size, hidden_size, action_size, self.device)

        elif self.model_type == 'CNN':
            input_shape = config['agent']['cnn']['input_shape']
            conv_channels = config['agent']['cnn']['conv_channels']
            hidden_size = config['agent']['cnn']['hidden_size']
            self.policy_net = CNN(input_shape, conv_channels, hidden_size, action_size, self.device)
            self.action_net = CNN(input_shape, conv_channels, hidden_size, action_size, self.device)
        else:
            raise ValueError("Unknown model type: ", self.model_type)

        #start with identical weights
        #action net's weights are updated less frequently for stability
        self.action_net.load_state_dict(self.policy_net.state_dict())
        self.action_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config['agent']['learning_rate']
        )
        
        self.memory = ReplayBuffer(config['memory']['buffer_size'])

        self.steps_done = 0

    
    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        #choose action based on epsilon greedy policy
        if evaluation_mode:
            epsilon = -1 #no exploration
        else:
            eps_start = self.config['agent']['epsilon_start']
            eps_end = self.config['agent']['epsilon_end']
            eps_decay = self.config['agent']['epsilon_decay']
            epsilon = eps_end + (eps_start-eps_end) * np.exp(-1. * self.steps_done / eps_decay)

        self.steps_done += 1

        if random.random() < epsilon:
            #exploration
            return random.randrange(self.action_size)
        else:
            #exploitation
            with torch.no_grad():
                #convert state into a tensor by adding batch dim
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                # highest action q value
                return q_values.argmax(dim=1).item()
            

    def update_action_net(self):
        #update action net weights to match policy net's
        self.action_net.load_state_dict(self.policy_net.state_dict())

    
    def learn(self):
        #samples batch from replay buffer and performs one learning step
        #the dqn optimization and loss is done here
        batch_size = self.config['memory']['batch_size']
        if len(self.memory) < batch_size:
            return
        
        experiences = self.memory.sample(batch_size)
        batch = Experience(*zip(*experiences))

        #we need to convert the batch data to tensors on correct device
        #we unsqueeze the done and reward tensors to [batch_size, 1]
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        #calculate q values for actions we actually took
        curr_q_values = self.policy_net(state_batch).gather(1, action_batch)

        #calculate target q values using DDQN
        with torch.no_grad():
            #select best action for next state
            next_actions = self.policy_net(next_state_batch).argmax(dim=1).unsqueeze(1)
            next_q_values = self.action_net(next_state_batch).gather(1, next_actions)
            #when done = 1, future value is 0
            target_q_values = reward_batch + (self.config['agent']['gamma'] * next_q_values * (1-done_batch))

        loss = F.mse_loss(curr_q_values, target_q_values)
        #clear prev gradients
        self.optimizer.zero_grad()
        loss.backward()

        #optinally clamp gradients to prevent explosion
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        self.optimizer.step() #update weights


    def save(self, file_name: str, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.policy_net.save(file_name, folder_path)
        print(f"\nAgent saved at path: {folder_path} as {file_name}")


    def load(self, file_name: str):
        self.policy_net.load(file_name)

        