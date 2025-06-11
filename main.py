import yaml
import numpy as np
import time

from src.environments.frozen_lake import FrozenLake
from src.agents.ddqn_agent import DDQNAgent
from src.training.trainer import Trainer

def train(config, render_mode=None):
    print("---Starting Training---")
    
    env = FrozenLake(
        config=config,
        render_mode=render_mode
    )

    agent = DDQNAgent(
        state_size=env.observation_space.shape[0], 
        action_size= env.action_space.n, 
        config=config
    )

    trainer = Trainer(
        agent=agent, 
        env=env, 
        config=config, 
        render_mode=render_mode,
        plot=False
    )

    trainer.run(
        policy_path="results/models",
        policy_name="policy.pth",
        plot_path="results/plots",
        plot_name="onehot.png",
        report_path="results/logs",
        report_name="onehot.md",
        shuffle_map=SHUFFLE_TRAIN_MAP
    )

    env.close()

def test(config, model_path: str):
    print("---Starting Testing---")

    env = FrozenLake(
        config=config,
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

    for i in range(num_test_episodes):
        state, _ = env.reset(shuffle_map=SHUFFLE_TEST_MAP)
        done = False
        episode_reward = 0
        print(f"\n---Starting test episode {i+1}/{num_test_episodes}---")
        env.render()
        time.sleep(1)

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            env.render()
            time.sleep(1/render_speed)
            done = terminated or truncated

        total_rewards.append(episode_reward)

    print(f"\nAverage score: {np.mean(total_rewards)}")
    env.close()

if __name__ == "__main__":
    MODE = 'train'
    RENDER_TRAINING = False
    SHUFFLE_TRAIN_MAP = True
    SHUFFLE_TEST_MAP = True
    CONFIG_PATH = 'configs/frozen_lake.yaml'
    MODEL_PATH = 'results/models/policy.pth'

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    if MODE == 'train':
        render_mode = 'human' if RENDER_TRAINING else None
        train(config, render_mode=render_mode)
    elif MODE == 'test':
        test(config, model_path=MODEL_PATH)
    else:
        print("Invalid mode. Choose 'train' or 'test'")
