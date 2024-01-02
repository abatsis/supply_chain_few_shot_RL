import sys
from pathlib import Path
import numpy as np
import pandas as pd
from normaliser import Normaliser
from lab import get_normalisation_data

means, stds = get_normalisation_data(env_size=74)


def group_by_max(list, reference_list):
    data = {'values': list, 'reference': reference_list}
    data_frame = pd.DataFrame(data)
    data_frame = data_frame.groupby(['reference']).max()
    return data_frame['values'].tolist()


def normalise(array, means, stds):
    result = array - means
    result = np.divide(result, stds, where=stds != 0)
    return result


def get_metalearner_mean_reward(file):
    file_name = Path(file).stem + '.csv'
    if 'skewed' in file:
        path = './data/evaluation/test_skewed/' + file_name
    else:
        path = './data/evaluation/test/' + file_name
    data_frame = pd.read_csv(path)
    values = data_frame['Metalearner'].values
    return np.mean(values)


def make_regrets(rewards, baseline_mean):
    get_regret = lambda reward: baseline_mean - reward
    return list(map(get_regret, rewards))


def load_data_from_file(file):
    data = np.load(file, allow_pickle=True)
    items = data.files

    ppo_mean_reward = data[items[4]].item()
    metalearner_mean_reward = get_metalearner_mean_reward(file)
    normaliser = Normaliser(file_name=file)

    file_generations = data[items[1]]
    file_online_rewards = data[items[0]]
    file_online_rewards = np.maximum.accumulate(group_by_max(file_online_rewards, file_generations))
    regrets = make_regrets(file_online_rewards, ppo_mean_reward)
    regrets_M = make_regrets(file_online_rewards, metalearner_mean_reward)
    file_online_rewards = normaliser(file_online_rewards)
    regrets = normaliser(regrets)
    regrets_M = normaliser(regrets_M)


    file_rewards = data[items[3]]
    file_rewards = normaliser(file_rewards)

    true_env, environments = data[items[5]], data[items[2]]
    true_env = normalise(true_env, means, stds)
    distance_from_true_env = lambda env: np.linalg.norm(env - true_env)
    distances = list(map(distance_from_true_env, environments))
    distances = np.maximum.accumulate(group_by_max(distances, file_generations))
    file_generations = group_by_max(file_generations, file_generations)

    return file_online_rewards, regrets, regrets_M, file_generations, file_rewards, distances
