import gymnasium as gym
import numpy as np

from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium import spaces
from src.utils.win_probability import compute_win_probability

class FrozenLake(gym.Wrapper):

    def __init__(self, config: dict, render_mode: str = None):
        self.map_size = config['env']['map_size']
        self.is_slippery = config['env']['is_slippery']
        self.render_md = render_mode
        self.config = config

        env = gym.make(
            'FrozenLake-v1',
            desc=generate_random_map(size=self.map_size),
            is_slippery=self.is_slippery,
            render_mode=render_mode
        )

        super().__init__(env)
        self.env = env
        self.state_size = self.observation_space.n

        #overwriting original state with a box space
        if self.config['agent']['model_type'] == 'MLP':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        elif self.config['agent']['model_type'] == 'CNN':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.config['agent']['cnn']['input_shape'][0], self.map_size, self.map_size), dtype=np.float32)
        else:
            raise ValueError("Unknown model type in yaml")

        self.prev_state = None
        self.win_prob = compute_win_probability(self.env.unwrapped.desc, self.map_size, self.is_slippery)
    

    def _cnn_state(self, s: int) -> np.ndarray:
        board = np.zeros((self.config['agent']['cnn']['input_shape'][0], self.map_size, self.map_size), dtype=np.float32)

        for i in range(self.map_size):
            for j in range(self.map_size):
                tile = self.env.unwrapped.desc[i][j]
                if tile == b'H':
                    board[2, i, j] = 1.0
                elif tile == b'G':
                    board[1, i, j] = 1.0
                else:
                    board[3, i, j] = 1.0

        row, col = divmod(s, self.map_size)
        board[0, row, col] = 1.0

        return board
    

    def _create_new_env(self):
        new_env = gym.make(
            'FrozenLake-v1',
            desc=generate_random_map(size=self.map_size),
            is_slippery=self.is_slippery,
            render_mode=self.render_md
        )
        self.env.close()
        self.env = new_env
        #updating state size in case it changed
        self.state_size = new_env.observation_space.n

        if self.config['agent']['model_type'] == 'MLP':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        elif self.config['agent']['model_type'] == 'CNN':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.config['agent']['cnn']['input_shape'][0], self.map_size, self.map_size), dtype=np.float32)
        else:
            raise ValueError("Unknown model type in yaml")

    
    def reset(self, shuffle_map: bool = False, **kwargs):
        if shuffle_map:
            self._create_new_env()

        #compute winning probability
        self.win_prob = compute_win_probability(self.env.unwrapped.desc, self.map_size, self.is_slippery)
        #print(f"[INFO] Probability of winning: {self.win_prob:.4f}")

        observation, info = self.env.reset(**kwargs)
        self.prev_state = observation

        if self.config['agent']['model_type'] == 'MLP':
            return self._full_state(observation), info
        elif self.config['agent']['model_type'] == 'CNN':
            return self._cnn_state(observation), info
        else:
            raise ValueError("Unknown model type in yaml")
    

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        old_state = self.prev_state

        hit_wall = (old_state == observation) and not terminated and not truncated
        
        if terminated and reward == 1.0:
            #print("reached goal")
            #agent reached goal
            reward = self.config['reward']['goal']
        elif terminated and reward == 0.0:
            #print("fell into hole")
            #fell into hole, so strong negative reward
            reward = self.config['reward']['hole']
        elif hit_wall:
            #print("hit wall")
            reward = self.config['reward']['wall']
        elif not terminated and not truncated:
            #print("moving")
            #small negative reward if agent just moves around ice
            reward = self.config['reward']['ice']
            
        self.prev_state = observation

        if self.config['agent']['model_type'] == 'MLP':
            observation = self._full_state(observation)
        elif self.config['agent']['model_type'] == 'CNN':
            observation = self._cnn_state(observation)
        else:
            raise ValueError("Unknown model type in yaml")
        
        return observation, reward, terminated, truncated, info
        

