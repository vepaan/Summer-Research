import yaml
import gymnasium as gym
import numpy as np
import os

from src.environments.frozen_lake import create_frozen_lake
from src.agents.ddqn_agent import DDQNAgent
from src.training.trainer import Trainer

def train(config, render_mode=None):
    print("---Starting Training---")
    
    env = create_frozen_lake(
        map_size=config['env']['map_size'],
        is_slippery=config['env']['is_slippery'],
        render_mode=render_mode
    )

    agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n, config)

    trainer = Trainer(agent, env, config, render_mode)
    trainer.run()
    env.close()

def test(config, model_path: str):
    print("---Starting Testing---")

    env = create_frozen_lake(
        map_size=config['env']['map_size'],
        is_slippery=config['env']['is_slippery'],
        render_mode='human'
    )

    agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n, config)
    try:
        agent.policy_net.load(model_path)
        print(f"Model loaded from path")
    except:
        print(f"Error: no model found at path")
        env.close()
        return
    
    render_speed = config['training']['speed']
    num_test_episodes = 10
    total_rewards = []


