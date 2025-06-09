import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), render_mode='human', map_name="8x8", is_slippery=True)
env.reset()

env.render()
