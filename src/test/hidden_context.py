import os
import secrets
import sys
from pathlib import Path
from lab import get_normalisation_data
from lab import save_explainability_data
import numpy as np
from lab import Lab
from tensorflow import keras
import cma

sys.path.append('src/models/')
from meta_learner import MetaLearner
from evaluate import load_model


def main():
    number_of_generations = 60
    number_of_offline_episodes = 20

    meta_learner = load_model()
    file_path = sys.argv[2]
    lab = Lab(input_sizes, output_size, max_number_of_levels)
    lab.read_environment(file_path)

    def get_reward(env_data):
        return lab.evaluate_metalearner(meta_learner, 1, env_data)

    # scaling
    means, stds = get_normalisation_data(lab.get_env_data_size())
    score = lambda x: get_reward(scale(x, means, stds))

    # CMA-ES Optimisation for env_data
    optimizer = cma.CMAEvolutionStrategy([0] * lab.get_env_data_size(), 1, inopts={'CMA_diagonal': False, })
    online_rewards = []
    generations = []
    environments = []
    for i in range(number_of_generations):
        solutions = optimizer.ask()
        environments.extend(solutions)
        values = []
        for solution in solutions:
            reward = score(solution)
            values.append(-reward)
            online_rewards.append(reward)
            generations.append(i + 1)
        print(np.mean(values), i)
        optimizer.tell(solutions, values)

    # off line evaluation after optimisation of context input
    env_data = np.mean(environments, axis=0)
    rewards, observations, actions = lab.evaluate_metalearner(meta_learner, number_of_offline_episodes,
                                                              env_data=scale(env_data, means, stds), logging=True)
    file_name = Path(file_path).stem
    destination_path = sys.argv[3]
    data_type = Path(destination_path).stem

    save_results(destination_path, file_name, online_rewards, generations, environments, rewards, lab)
    save_explainability_data(observations, actions, lab.env_config(), rewards, f'data/explainability/{data_type}',
                             file_name)


def scale(point, means, stds): return (stds * point) + means


def save_results(destination_path, file_name, online_rewards, generations, environments, rewards, lab):
    os.makedirs(destination_path, exist_ok=True)
    file = destination_path + "/" + file_name + ".npz"
    np.savez(file, online_rewards=online_rewards, generations=generations, environments=environments, rewards=rewards,
             ppo_mean_reward=lab.get_mean_ppo_reward(), true_env=lab.true_env_to_vector())


input_sizes = [170, 11, 11, 1, 1, 11, 1, 2, 11, 11, 1, 1, 11, 1]
output_size = 11
max_number_of_levels = 9

if __name__ == "__main__":
    main()
