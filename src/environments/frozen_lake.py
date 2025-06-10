import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import keyboard

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), render_mode='human', map_name="8x8", is_slippery=True)
env.reset()
env.render()
episode_over = False

while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
    if keyboard.is_pressed('q'):
        env.close()

