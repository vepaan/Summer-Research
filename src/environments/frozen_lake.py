import gymnasium as gym
import numpy as np

from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium import spaces
from src.utils.win_probability import compute_win_probability, approximate_win_probability

class FrozenLake(gym.Wrapper):

    def __init__(self, config: dict, render_mode: str = None, compute_win_prob: bool = False):
        self.map_size = config['env']['map_size']
        self.render_md = render_mode
        self.config = config

        env = gym.make(
            'FrozenLake-v1',
            desc=generate_random_map(size=self.map_size),
            is_slippery=False,
            render_mode=render_mode,
            max_episode_steps=config['training']['max_steps_per_episode']
        )

        super().__init__(env)
        self.env = env
        self.state_size = self.observation_space.n
        self.agent_pos = (0, 0)
        #print(self.agent_pos)

        #overwriting original state with a box space
        if self.config['agent']['model_type'] == 'MLP':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        elif self.config['agent']['model_type'] == 'CNN':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.config['agent']['cnn']['input_shape'][0], self.map_size, self.map_size), dtype=np.float32)
        else:
            raise ValueError("Unknown model type in yaml")

        self.prev_state = None
        self.compute_win_prob = compute_win_prob
        if self.compute_win_prob:
            self.win_prob = approximate_win_probability(self.env.unwrapped.desc, self.config['env']['slip'])
        else:
            self.win_prob = -1


    def apply_slip(self, intended_action: int) -> int:
        slip_probs = self.config['env']['slip']
        assert abs(sum(slip_probs) - 1.0) < 1e-5, "Slip probabilities must sum to 1"

        # Define the 3 possible actions: forward, left, right relative to intended
        slip_deltas = {
            0: [0, 3, 1],  # LEFT → [LEFT, UP, DOWN]
            1: [1, 0, 2],  # DOWN → [DOWN, LEFT, RIGHT]
            2: [2, 1, 3],  # RIGHT → [RIGHT, DOWN, UP]
            3: [3, 2, 0],  # UP → [UP, RIGHT, LEFT]
        }

        if slip_probs[0] == 1.0:
            return intended_action  # deterministic

        slip_actions = slip_deltas[intended_action]
        return np.random.choice(slip_actions, p=slip_probs)


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
    

    def _partial_cnn_state(self) -> np.ndarray:
        channels = self.config['agent']['cnn']['input_shape'][0]
        local_view = np.zeros((channels, 3, 3), dtype=np.float32)
        board = self.env.unwrapped.desc
        i, j = self.agent_pos

        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.map_size and 0 <= nj < self.map_size:
                    tile = board[ni][nj]
                    if tile == b'H':
                        local_view[2, di+1, dj+1] = 1.0
                    elif tile == b'G':
                        local_view[1, di+1, dj+1] = 1.0
                    else:
                        local_view[3, di+1, dj+1] = 1.0

        # Always mark the center (agent’s own position)
        local_view[0, 1, 1] = 1.0
        return local_view
    

    def is_action_safe(self, pos: tuple, action: int) -> bool:
        i, j = pos
        size = self.map_size
        desc = self.env.unwrapped.desc

        slip_probs = self.config['env']['slip']  # Default to deterministic
        is_deterministic = slip_probs[0] == 1.0

        # Action-to-direction mapping
        move = {
            0: (0, -1),   # LEFT
            1: (1, 0),    # DOWN
            2: (0, 1),    # RIGHT
            3: (-1, 0),   # UP
        }

        # If deterministic, only check the intended direction
        if is_deterministic:
            dx, dy = move[action]
            ni, nj = i + dx, j + dy
            if 0 <= ni < size and 0 <= nj < size:
                return desc[ni][nj] != b'H'
            return True  # Treat out-of-bounds as safe

        # Otherwise (stochastic), check all possible slip outcomes
        slip_deltas = {
            0: [move[0], move[3], move[1]],  # LEFT → [LEFT, UP, DOWN]
            1: [move[1], move[0], move[2]],  # DOWN → [DOWN, LEFT, RIGHT]
            2: [move[2], move[1], move[3]],  # RIGHT → [RIGHT, DOWN, UP]
            3: [move[3], move[2], move[0]],  # UP → [UP, RIGHT, LEFT]
        }

        for dx, dy in slip_deltas[action]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < size and 0 <= nj < size:
                if desc[ni][nj] == b'H':
                    return False
        return True


    def _create_new_env(self):
        new_env = gym.make(
            'FrozenLake-v1',
            desc=generate_random_map(size=self.map_size),
            is_slippery=False,
            render_mode=self.render_md,
            max_episode_steps=self.config['training']['max_steps_per_episode']
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
        if self.compute_win_prob:
            self.win_prob = approximate_win_probability(self.env.unwrapped.desc, self.config['env']['slip'])
        else:
            self.win_prob = -1
        #print(f"[INFO] Probability of winning: {self.win_prob:.4f}")

        observation, info = self.env.reset(**kwargs)
        self.prev_state = observation
        self.agent_pos = (observation // self.map_size, observation % self.map_size)
        #print(self.agent_pos)

        if self.config['agent']['model_type'] == 'MLP':
            return self._full_state(observation), info
        elif self.config['agent']['model_type'] == 'CNN':
            return self._cnn_state(observation), info
            #return self._partial_cnn_state(), info
        else:
            raise ValueError("Unknown model type in yaml")
    

    def step(self, action):
        action = self.apply_slip(action)
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

        self.agent_pos = (observation // self.map_size, observation % self.map_size)
        #print(self.agent_pos)

        if self.config['agent']['model_type'] == 'MLP':
            observation = self._full_state(observation)
        elif self.config['agent']['model_type'] == 'CNN':
            observation = self._cnn_state(observation)
            #observation = self._partial_cnn_state()
        else:
            raise ValueError("Unknown model type in yaml")

        return observation, reward, terminated, truncated, info
        

