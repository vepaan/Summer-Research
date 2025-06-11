import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium import spaces
import numpy as np

class FrozenLake(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.state_size = self.observation_space.n
        #overwriting original state with a box space for one hot vector
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.prev_state = None
        
    def _to_one_hot(self, s: int) -> np.ndarray:
        one_hot = np.zeros(self.state_size, dtype=np.float32)
        one_hot[s] = 1.0
        return one_hot
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._to_one_hot(observation), info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        old_state = self.prev_state

        hit_wall = (old_state == observation) and not terminated and not truncated
        
        if terminated and reward == 1.0:
            #print("reached goal")
            #agent reached goal
            pass
        elif terminated and reward == 0.0:
            #print("fell into hole")
            #fell into hole, so strong negative reward
            reward = -1.0
        elif hit_wall:
            #print("hit wall")
            reward = -0.1
        elif not terminated and not truncated:
            #print("moving")
            #small negative reward if agent just moves around ice
            reward = -0.01
            
        self.prev_state = observation
        return self._to_one_hot(observation), reward, terminated, truncated, info
        
    
def create_frozen_lake(map_size: int = 4, is_slippery: bool = False, render_mode: str = None) -> gym.Env:
    random_map = generate_random_map(size=map_size)
    env = gym.make(
        'FrozenLake-v1',
        desc=random_map,
        is_slippery=is_slippery,
        render_mode = render_mode
    )
    wrapped_env = FrozenLake(env)
    return wrapped_env

