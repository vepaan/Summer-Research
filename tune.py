import yaml
import numpy as np
from functools import partial
import optuna

from src.environments.frozen_lake import FrozenLake
from src.agents.ddqn_agent import DDQNAgent
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import Trainer


def objective(trial: optuna.Trial, base_config: dict) -> float:
    #This is the objective function that Optuna will minimize or maximize.
    #trial (optuna.Trial): A trial object from Optuna used to suggest hyperparameters.
    #base_config (dict): The original base configuration from the YAML file.
    
    trial_config = base_config.copy()

    trial_config['agent']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    trial_config['agent']['gamma'] = trial.suggest_categorical("gamma", [0.95, 0.99, 0.995])
    trial_config['agent']['epsilon_decay'] = trial.suggest_float("epsilon_decay", 1000, 10000)
    trial_config['agent']['target_update_freq'] = trial.suggest_categorical("target_update_freq", [10, 50, 100])
    
    trial_config['agent']['cnn']['hidden_size'] = trial.suggest_categorical("cnn_hidden_size", [64, 128, 256])
    trial_config['agent']['cnn']['conv_channels'] = trial.suggest_categorical("cnn_conv_channels", [16, 32])
    
    trial_config['memory']['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    trial_config['training']['max_steps_per_episode'] = trial.suggest_categorical("max_steps_per_episode", [100, 200, 300])

    trial_config['reward']['hole'] = trial.suggest_float("reward_hole", -1.0, 0.0)
    trial_config['reward']['wall'] = trial.suggest_float("reward_wall", -0.5, 0.0)
    trial_config['reward']['ice'] = trial.suggest_float("reward_ice", -0.05, 0.05)

    #now start training
    env = FrozenLake(config=trial_config, render_mode=None)

    state_size = env.observation_space.shape

    if len(state_size) == 3:
        #CNN expectes (C, H, W)
        pass
    else:
        state_size = state_size[0]

    agent = DDQNAgent(
        state_size=state_size, 
        action_size=env.action_space.n, 
        config=trial_config
    )

    trainer = Trainer(
        agent=agent, 
        env=env, 
        config=trial_config, 
        render_mode=None, 
        plot=False
    )

    trainer.run(shuffle_map=True)

    final_avg_score = np.mean(trainer.scores_window) if trainer.scores_window else float('-inf')
    env.close()

    #optuna optimizes this
    return final_avg_score


if __name__ == "__main__":

    CONFIG_PATH = 'configs/frozen_lake.yaml'
    with open(CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    #create a study object where we want to maximize the score.
    study = optuna.create_study(direction="maximize")

    #run the optimization.
    #'n_trials' is the number of different hyperparameter combinations to try.
    #'partial' is used to pass the static 'base_config' to our objective function.
    study.optimize(partial(objective, base_config=base_config), n_trials=50)

    best_trial = study.best_trial

    print("\n--- Hyperparameter Tuning Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial's final score: {best_trial.value:.4f}")
    
    print("\nBest hyperparameters found were: ")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")