import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os

from src.models.networks import DQN
from src.utils.replay_buffer import ReplayBuffer, Experience

class DDQNAgent:

    def __init__(self, state_size: int, action_size: int, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"DDQN Agent is using device: {self.device}")

        self.policy_net = DQN(state_size, config['agent']['hidden_size'], action_size, self.device)
        
        #action net's weights are updated less frequently for stability
        self.action_net = DQN(state_size, config['agent']['hidden_size'], action_size, self.device)
        #start with identical weights
        self.action_net.load_state_dict(self.policy_net.state_dict())
        self.action_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['agent']['learning_rate'])
        self.memory = ReplayBuffer(config['memory']['buffer_size'])

        self.steps_done = 0

    
    def act(self, state: np.ndarray) -> int:
        #choose action based on epsilon greedy policy
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
        pass


    def save(self, file_name: str = "model.pth", folder_path: str = "../../results/models"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        full_path = os.path.join(folder_path, file_name)
        self.policy_net.save(full_path)

        