import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Set the paths to your two CSV files ---
# IMPORTANT: Replace these placeholder paths with the actual paths to your data files.
csv_path_1 = './results/data/ddqn_cnn_slippery.csv'  # Replace with the path to your first file
csv_path_2 = './results/data/ppo_cnn_slippery.csv'   # Replace with the path to your second file

# --- 2. Load the data using pandas ---
try:
    # Read the data from the CSV files into pandas DataFrames
    data1 = pd.read_csv(csv_path_1)
    data2 = pd.read_csv(csv_path_2)

    # Extract clean filenames to use in the plot legend
    file_name_1 = os.path.basename(csv_path_1).replace('_data.csv', '')
    file_name_2 = os.path.basename(csv_path_2).replace('_data.csv', '')

    print(f"Successfully loaded data for '{file_name_1}' and '{file_name_2}'.")

    # --- 3. Create the comparison plot ---
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 7))

    # Plot the '100-Ep Rolling Avg' from the first file
    plt.plot(data1['Episode'], data1['100-Ep Rolling Avg'], label=f'{file_name_1} (Rolling Avg)')

    # Plot the '100-Ep Rolling Avg' from the second file on the same axes
    plt.plot(data2['Episode'], data2['100-Ep Rolling Avg'], label=f'{file_name_2} (Rolling Avg)')

    # --- 4. Customize and display the chart ---
    plt.title('Comparison of Learning Progress', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('100-Episode Rolling Average Reward', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Optional: Save the comparison chart to a file
    plt.savefig('learning_curve_comparison.png', dpi=300)
    print("Comparison chart saved as 'learning_curve_comparison.png'")

    # Show the plot on the screen
    plt.show()


except FileNotFoundError as e:
    print(f"Error: Could not find a file.")
    print(f"Please make sure the path '{e.filename}' is correct and the file exists in that location.")
except KeyError as e:
    print(f"Error: A required column {e} was not found in one of the CSV files.")
    print("Please ensure your CSVs contain the 'Episode' and '100-Ep Rolling Avg' columns.")