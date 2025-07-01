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


def plot_test_results(df: pd.DataFrame):
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 1. Win rate per difficulty tier
    win_rate_by_tier = df.groupby("tier")["win"].mean().reindex(["easy", "medium", "hard"])

    plt.figure(figsize=(6, 4))
    sns.barplot(x=win_rate_by_tier.index, y=win_rate_by_tier.values, palette="Blues_d")
    plt.title("Win Rate by Difficulty Tier")
    plt.ylabel("Win Rate")
    plt.xlabel("Difficulty Tier")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # 2. Reward distribution by tier
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="tier", y="reward", palette="Set3", order=["easy", "medium", "hard"])
    plt.title("Reward Distribution by Difficulty Tier")
    plt.tight_layout()
    plt.show()

    # 3. Difficulty vs win scatter
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="difficulty", y="win", alpha=0.6, edgecolor=None)
    plt.title("Win Outcomes vs Map Difficulty")
    plt.xlabel("Map Difficulty Score")
    plt.ylabel("Win (1=True, 0=False)")
    plt.tight_layout()
    plt.show()

    # 4. Difficulty vs reward scatter + line of best fit
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x="difficulty", y="reward", scatter_kws={'alpha': 0.5})
    plt.title("Reward vs Map Difficulty")
    plt.tight_layout()
    plt.show()

    # 5. Win count per tier
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="tier", hue="win", palette="Set2", order=["easy", "medium", "hard"])
    plt.title("Win/Loss Count by Difficulty Tier")
    plt.xlabel("Difficulty Tier")
    plt.ylabel("Number of Episodes")
    plt.legend(title="Win", labels=["Loss", "Win"])
    plt.tight_layout()
    plt.show()

    # 6. KDE Plot of Difficulty
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df[df["win"] == 1]["difficulty"], label="Wins", shade=True)
    sns.kdeplot(data=df[df["win"] == 0]["difficulty"], label="Losses", shade=True)
    plt.title("Density of Difficulty for Wins vs Losses")
    plt.xlabel("Win Prob")
    plt.legend()
    plt.tight_layout()
    plt.show()



