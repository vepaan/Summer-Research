import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os

from src.models.networks import TransformerNetwork
from src.utils.replay_buffer import ReplayBuffer, Experience

class DDQNTransformerAgent:

    def __init__(self, state_size: int, action_size: int, config: dict, env=None, shield: bool = False):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.env = env
        self.shield = shield

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"DDQN Agent is using device: {self.device}")

        # Hyperparams for transformer input
        self.seq_len = config['agent']['transformer']['seq_len']
        self.cnn_channels = config['agent']['cnn']['input_shape'][0]
        self.frame_size = 3 * 3  # since 3x3 view
        self.flattened_dim = self.cnn_channels * self.frame_size

        self.policy_net = TransformerNetwork(
            input_dim=self.flattened_dim,
            seq_len=self.seq_len,
            d_model=config['agent']['transformer']['d_model'],
            num_heads=config['agent']['transformer']['num_heads'],
            hidden_dim=config['agent']['transformer']['hidden_dim'],
            num_actions=self.action_size,
            device=self.device
        )

        self.action_net = TransformerNetwork(
            input_dim=self.flattened_dim,
            seq_len=self.seq_len,
            d_model=config['agent']['transformer']['d_model'],
            num_heads=config['agent']['transformer']['num_heads'],
            hidden_dim=config['agent']['transformer']['hidden_dim'],
            num_actions=self.action_size,
            device=self.device
        )

        #start with identical weights
        #action net's weights are updated less frequently for stability
        self.action_net.load_state_dict(self.policy_net.state_dict())
        self.action_net.eval()

        #for cuda purposes
        self.policy_net = self.policy_net.to(self.device)
        self.action_net = self.action_net.to(self.device)

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config['agent']['learning_rate_ddqn']
        )
        
        self.memory = ReplayBuffer(config['memory']['buffer_size'])
        self.steps_done = 0


    def _process_sequence(self, state: np.ndarray):
        obs_tensor = torch.tensor(np.array(state), dtype=torch.float32) # (T, C, 3, 3)
        obs_tensor = obs_tensor.view(len(state), -1) # (T, C*3*3)
        return obs_tensor.unsqueeze(0).to(self.device) # (1, T, D)

    
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
        
        #exploitation
        with torch.no_grad():
            #convert state into a tensor by adding batch dim
            state_tensor = self._process_sequence(state)
            q_values = self.policy_net(state_tensor) # (1, num_actions)
            ranked_actions = torch.argsort(q_values[0], descending=True).tolist()

        #shielding
        if self.env and self.shield:
            for action in ranked_actions:
                if self.env.is_action_safe(self.env.agent_pos, action):
                    #highest q value action that is also safe
                    return action
                
        #fall back in case no actions are safe
        return ranked_actions[0]
            
            
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

        def to_seq_tensor(seq_batch):
            batch_tensor = torch.tensor(np.array(seq_batch), dtype=torch.float32)  # (B, T, C, 3, 3)
            B, T, C, H, W = batch_tensor.shape
            return batch_tensor.view(B, T, C * H * W).to(self.device)

        #we need to convert the batch data to tensors on correct device
        #we unsqueeze the done and reward tensors to [batch_size, 1]
        state_batch = to_seq_tensor(batch.state)
        next_state_batch = to_seq_tensor(batch.next_state)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        #calculate q values for actions we actually took
        curr_q_values = self.policy_net(state_batch).gather(1, action_batch)

        #calculate target q values using DDQN
        with torch.no_grad():
            #select best action for next state
            next_actions = self.policy_net(next_state_batch).argmax(dim=1).unsqueeze(1)
            next_q_values = self.action_net(next_state_batch).gather(1, next_actions)
            #when done = 1, future value is 0
            target_q_values = reward_batch + (self.config['agent']['gamma_ddqn'] * next_q_values * (1-done_batch))

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
        self.policy_net = self.policy_net.to(self.device)
        self.action_net = self.action_net.to(self.device)

        