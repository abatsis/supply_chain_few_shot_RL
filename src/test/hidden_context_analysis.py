import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from load_cma_data import load_data_from_file


def load_data(folder):
    path = "./data/evaluation/"
    path = path + folder + '/'
    files = [path + file for file in os.listdir(path) if not file.startswith('.')]
    regret = []
    regret_M = []
    online_rewards = []
    generations = []
    rewards = []
    true_environment_distances = []
    for file in files:
        file_online_rewards, regrets, regrets_M, file_generations, file_rewards, distances = load_data_from_file(file)
        online_rewards.extend(file_online_rewards)
        regret.extend(regrets)
        regret_M.extend(regrets_M)
        generations.extend(file_generations)
        rewards.extend(file_rewards)
        true_environment_distances.extend(distances)

    return online_rewards, generations, rewards, regret, regret_M, true_environment_distances


data_type = sys.argv[1]
os.makedirs('reports/evaluation_hidden', exist_ok=True)
online_rewards, generations, rewards, regret, regret_M, true_environment_distances = load_data(data_type)

sns.set()
data = {'reward': online_rewards, 'generation': generations, 'distance': true_environment_distances}
regret_data = {'generation': generations, 'PPO': regret} if 'skewed' in data_type else {'generation': generations,
                                                                                        'PPO': regret,
                                                                                        'Metalearner': regret_M}
data_frame = pd.DataFrame(data)
regret_data_frame = pd.DataFrame(regret_data)

graph = sns.lineplot(x="generation", y="reward", estimator=np.mean, data=data_frame)
plt.savefig(f"./reports/evaluation_hidden/online_rewards_{data_type}.pdf", dpi=300)
plt.clf()

graph = sns.lineplot(x='generation', y='regret', hue='baseline',
                     data=pd.melt(regret_data_frame, ['generation'], value_name='regret', var_name='baseline'))
plt.savefig(f"./reports/evaluation_hidden/online_regret_{data_type}.pdf", dpi=300)
plt.clf()

graph = sns.lineplot(x="generation", y="distance", estimator=np.mean, data=data_frame)
plt.savefig(f"./reports/evaluation_hidden/env_distances_{data_type}.pdf", dpi=300)

# What to do with offline rewards??????
