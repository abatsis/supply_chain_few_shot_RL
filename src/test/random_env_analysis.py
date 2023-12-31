import os
import random
import shutil
import sys
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from load_cma_data import load_data_from_file

data_type = sys.argv[1]


def get_env_configuration(file):
    file_name = Path(file).stem
    path = f'./data/{data_type}/{file_name}.npz'
    data = np.load(path, allow_pickle=True)
    files = data.files
    item = files[-1]
    return data[item].item()

def make_dir(file):
    file_name = Path(file).stem
    path = f'./reports/random_env_{data_type}/' + file_name
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path + '/'


path = f"./data/evaluation/{data_type}_hidden_context/"
files = [path + file for file in os.listdir(path) if not file.startswith('.')]
file = random.choice(files)
destination_path = make_dir(file)
online_rewards, regret, regret_M, generations, rewards, true_environment_distances = load_data_from_file(file)

sns.set()
data = {'reward': online_rewards, 'generation': generations, 'distance': true_environment_distances}
regret_data = {'generation': generations, 'PPO': regret, 'Metalearner': regret_M}
data_frame = pd.DataFrame(data)
regret_data_frame = pd.DataFrame(regret_data)

graph = sns.lineplot(x="generation", y="reward", estimator=np.mean, data=data_frame)
plt.savefig(f'{destination_path}online_rewards.pdf', dpi=300)
plt.clf()

graph = sns.lineplot(x='generation', y='regret', hue='baseline',
                     data=pd.melt(regret_data_frame, ['generation'], value_name='regret', var_name='baseline'))
plt.savefig(f'{destination_path}online_regret.pdf', dpi=300)
plt.clf()

graph = sns.lineplot(x="generation", y="distance", estimator=np.mean, data=data_frame)
plt.savefig(f'{destination_path}env_distances.pdf', dpi=300)

with open(destination_path + 'env', 'w') as f:
    print(get_env_configuration(file), file=f)
