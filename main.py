import yaml
import numpy as np
import time
import pandas as pd
from tqdm import tqdm

from src.environments.frozen_lake import FrozenLake
from src.agents.ddqn_agent import DDQNAgent
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import DDQNTrainer, PPOTrainer


def _load_config_from_log(log_path: str) -> dict:
    with open(log_path, 'r') as f:
        lines = f.readlines()

    start, end = None, None
    for idx, line in enumerate(lines):
        if line.strip() == '```yaml':
            start = idx + 1
        elif line.strip() == '```' and start is not None:
            end = idx
            break

    if start is None or end is None:
        raise ValueError("YAML block not found in log file")
    
    yaml_block = ''.join(lines[start:end])
    config = yaml.safe_load(yaml_block)

    return config


def train(config, render_mode=None):
    print("---Starting Training---")
    
    env = FrozenLake(
        config=config,
        render_mode=render_mode
    )

    if config['agent']['rl_type'].lower() == 'ddqn':

        agent = DDQNAgent(
            state_size=env.observation_space.shape[0], 
            action_size=env.action_space.n, 
            config=config,
            env=env
        )

        trainer = DDQNTrainer(
            agent=agent, 
            env=env, 
            config=config, 
            render_mode=render_mode,
            plot=False
        )

    elif config['agent']['rl_type'].lower() == 'ppo':

        agent = PPOAgent(
            state_size=env.observation_space.shape[0], 
            action_size=env.action_space.n, 
            config=config
        )

        trainer = PPOTrainer(
            agent=agent,
            env=env,
            config=config,
            render_mode=render_mode,
            plot=False
        )

    else:
        raise ValueError("Unknown RL algorithm in train")


    trainer.run(
        policy_path="results/models",
        policy_name=f"{APPROACH}.pth",
        plot_path="results/plots",
        plot_name=f"{APPROACH}.png",
        data_path="results/data",
        data_name=f"{APPROACH}.csv",
        report_path="results/logs",
        report_name=f"{APPROACH}.md",
        shuffle_map=SHUFFLE_TRAIN_MAP
    )

    env.close()


def test(config, model_path: str):
    print("---Starting Testing---")

    env = FrozenLake(
        config=config,
        render_mode='human' if RENDER_TESTING else None
    )
    
    if config['agent']['rl_type'].lower() == 'ddqn':
        agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n, config, env)
    elif config['agent']['rl_type'].lower() == 'ppo':
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, config)
    else:
        raise ValueError("Unknown RL algorithm in test")

    try:
        agent.load(model_path)
        print(f"Model loaded from path")
    except:
        print(f"Error: no model found at path")
        env.close()
        return
    
    render_speed = config['testing']['speed']
    num_test_episodes = config['testing']['num_episodes']
    test_results = []
    total_rewards = []
    wins = 0

    for _ in tqdm(range(num_test_episodes), desc="Testing Episodes"):
        state, _ = env.reset(shuffle_map=SHUFFLE_TEST_MAP)
        map_difficulty = env.win_prob
        won = False
        done = False
        episode_reward = 0
        
        if RENDER_TESTING:
            env.render()
            time.sleep(1)

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward

            if RENDER_TESTING:
                env.render()
                time.sleep(1/render_speed)

            done = terminated or truncated
            if reward == config['reward']['goal']:
                won = True
                wins += 1

        total_rewards.append(episode_reward)
        test_results.append((won, map_difficulty))


    df = pd.DataFrame(test_results, columns=['win', 'difficulty'])
    
    q1 = df['difficulty'].quantile(1/3)
    q2 = df['difficulty'].quantile(2/3)

    def categorize(d: float) -> str:
        if d <= q1:
            return "hard"
        elif d <= q2:
            return "medium"
        else:
            return "easy"

    df["tier"] = df["difficulty"].apply(categorize)

    #TEST ANALYTICS
    print(f"\nAverage score: {np.mean(total_rewards)}\n")
    print(f"Overall win rate: {wins}/{num_test_episodes}: {wins/num_test_episodes}\n")

    for tier in ["easy", "medium", "hard"]:
        tier_df = df[df["tier"] == tier]
        tier_win_rate = tier_df["win"].mean()
        print(f"{tier.capitalize()} Maps:")
        print(f"  Count: {len(tier_df)}")
        print(f"  Win Rate: {tier_win_rate:.3f}")

    env.close()

if __name__ == "__main__":

    APPROACH = 'ddqn_cnn_6x6_slippery'
    MODE = 'train'

    RENDER_TRAINING = False
    RENDER_TESTING = False

    SHUFFLE_TRAIN_MAP = True
    SHUFFLE_TEST_MAP = True

    CONFIG_PATH = 'configs/frozen_lake.yaml'
    MODEL_PATH = f'results/models/{APPROACH}.pth'
    LOG_PATH = f'results/logs/{APPROACH}.md'

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    if MODE == 'train':
        render_mode = 'human' if RENDER_TRAINING else None
        train(config, render_mode=render_mode)
    elif MODE == 'test':
        test(_load_config_from_log(log_path=LOG_PATH), model_path=MODEL_PATH)
    else:
        print("Invalid mode. Choose 'train' or 'test'")
