import yaml
import gymnasium as gym
import numpy as np
import time
import argparse

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

    #modify trainer if rendering enabled
    if render_mode == 'human':
        original_run = Trainer.run
        