import os
import sys
from normaliser import Normaliser
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def main():
    data_type = sys.argv[1]
    os.makedirs('reports/evaluation', exist_ok=True)

    columns = ['Mean value', 'Standard', 'Min', 'Quartile 1', 'Quartile 2', 'Quartile 3', 'Max']
    rows = ['PPO', 'Meta-learner']
    sns.set()

    data_frame = load_data(data_type)
    table_frame = pd.DataFrame(get_stats(data_frame), rows, columns)
    with open(f'./reports/evaluation/analysis_report_{data_type}.txt', 'w') as f:
        print(table_frame, file=f)
        print(table_frame.to_latex(), file=f)

    difference_frame = pd.DataFrame()
    difference_frame['Reward difference'] = data_frame['Metalearner'] - data_frame['PPO']
    sns.violinplot(data=difference_frame, y="Reward difference")
    plt.savefig(f"./reports/evaluation/reward_difference_{data_type}.pdf", dpi=300)
    plt.clf()

    models = ['PPO', 'Metalearner']
    data_frame = pd.melt(data_frame, value_vars=models, var_name='Model', value_name='reward', ignore_index=False)
    sns.violinplot(data=data_frame, y="reward", x="Model")
    plt.savefig(f"./reports/evaluation/ppo_vs_metalearner_{data_type}.pdf", dpi=300)


def array_stats(array):
    stats = [np.mean(array), np.std(array)]
    stats = stats + [np.quantile(array, i * 0.25) for i in range(5)]
    return stats


def get_stats(data_frame):
    rl_rewards = data_frame['PPO'].values
    nn_rewards = data_frame['Metalearner'].values
    return [array_stats(rl_rewards), array_stats(nn_rewards)]


def load_data(folder):
    path = "./data/evaluation/"
    path = path + folder + '/'
    files = [path + file for file in os.listdir(path) if not file.startswith('.')]
    data_frames = []
    for file in files:
        data_frame = pd.read_csv(file)
        normaliser = Normaliser(data_frame['RL'].values)
        data_frame['RL'] = normaliser(data_frame['RL'].values)
        data_frame['Metalearner'] = normaliser(data_frame['Metalearner'].values)
        data_frames.append(data_frame)
    data_frame = pd.concat(data_frames)
    data_frame.rename(columns={'RL': 'PPO'}, inplace=True)
    return data_frame


if __name__ == "__main__":
    main()