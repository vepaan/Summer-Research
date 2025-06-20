import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

class LivePlotter:

    def __init__(self, title="Learning Progress", xlabel='Episode', ylabel='Total Reward'):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        sns.set_theme(style='darkgrid')
        self.scores = []
        self.rolling_avg = []

        self.ax.plot([], [], label='Episode Score', color='orange')
        self.ax.plot([], [], label='100-Ep Rolling Avg', color='gray', alpha=0.4)

        self.ax.legend()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)


    def update(self, scores: list):
        self.scores = scores
        rolling_avg_series = pd.Series(self.scores).rolling(window=100).mean()
        self.rolling_avg = rolling_avg_series.tolist()

        score_line = self.ax.lines[0]
        avg_line = self.ax.lines[1]

        score_line.set_data(np.arange(len(self.scores)), self.scores)
        avg_line.set_data(np.arange(len(self.rolling_avg)), self.rolling_avg)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


    def save(self, folder_path: str, file_name: str):
        plot_dir = os.path.dirname(folder_path)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.fig.savefig(file_name)
        plt.close(self.fig)
        plt.ioff()
        print("\nPlot saved")
        


def plot_rewards(scores: list, folder_path: str, file_name: str, data_path: str, data_name: str):
    #plot per ep scores and a 100 episode rolling avg
    print(f"\nPlotting the rewards against eps")

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 6))

    df = pd.DataFrame({'Episode Score': scores})
    df['100-Ep Rolling Avg'] = df['Episode Score'].rolling(window=100).mean()

    plt.plot(df['Episode Score'], label='Episode Score', color='orange')
    plt.plot(df['100-Ep Rolling Avg'], label='100-Ep Rolling Avg', color='gray', alpha=0.7)

    plt.title('Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, file_name))

    data_save_path = os.path.join(data_path, data_name)
    df.to_csv(data_save_path, index_label='Episode')

    print(f"Data saved to {data_save_path}")

    plt.close()
    print("\nPlot saved")

