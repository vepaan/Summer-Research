from collections import deque
import numpy as np
from tqdm import tqdm
import time

class Trainer:

    def __init__(self, agent, env, config: dict, render_mode: str = None):
        self.agent = agent
        self.env = env
        self.config = config
        self.render_mode = render_mode

        self.num_episodes = config['training']['num_episodes']
        self.max_steps_per_episode = config['training']['max_steps_per_episode']
        self.log_interval = config['training']['log_interval']
        self.target_update_freq = config['agent']['target_update_freq']
        self.save_interval = config['training']['save_interval']

        self.render_speed = config['training']['speed']

        #for logging and tracking
        self.scores = []
        self.scores_window = deque(maxlen=100)

    
    def run(self, save_path: str, save_name: str):
        #we use tqdm for a clean progress bar over episodes
        for i_episode in tqdm(range(1, self.num_episodes+1), desc="Training Episodes"):
            state, info = self.env.reset(shuffle_map=True)
            score = 0

            if self.render_mode == 'human':
                self.env.render()
                time.sleep(1/self.render_speed)

            for _ in range(self.max_steps_per_episode):
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                if self.render_mode == 'human':
                    self.env.render()
                    time.sleep(1/self.render_speed)

                done = terminated or truncated
                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.learn()
                state = next_state
                score += reward
                if done:
                    break

            self.scores_window.append(score)
            self.scores.append(score)

            if i_episode % self.log_interval == 0:
                avg = np.mean(self.scores_window)
                print(f'\rEpisode {i_episode}\tAverage Score (last 100): {avg:.2f}')

            #periodically updaing and saving
            if i_episode % self.target_update_freq == 0:
                self.agent.update_action_net()
            if i_episode % self.save_interval == 0:
                self.agent.save(file_name=save_name, folder_path=save_path)

        print("\nTraining Finished")
        print(f'Final average score of last 100 eps: {np.mean(self.scores_window):.2f}')
        self.agent.save(file_name=save_name, folder_path=save_path)