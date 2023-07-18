import secrets
import sys
from pathlib import Path
from lab import get_normalisation_data
import numpy as np
from lab import Lab
from tensorflow import keras
import cma

sys.path.append('src/models/')
from meta_learner import MetaLearner

input_sizes = [170, 11, 11, 1, 1, 11, 1, 2, 11, 11, 1, 1, 11, 1]
output_size = 11
lab = Lab(input_sizes, output_size)
number_of_generations = 60
number_of_offline_episodes = 20
model = keras.models.load_model(sys.argv[1])
meta_learner = MetaLearner(input_sizes, output_size)
meta_learner.set_model(model)


def get_reward(env_data): return lab.evaluate_metalearner(meta_learner, 1, env_data)




def scale(point, means, stds): return (stds * point) + means


file_path = sys.argv[2]
lab.read_environment(file_path)

# scaling
means, stds = get_normalisation_data(lab.get_env_data_size())
score = lambda x: get_reward(scale(x, means, stds))
optimizer = cma.CMAEvolutionStrategy([0] * lab.get_env_data_size(), 1, inopts={'CMA_diagonal': False,})

# CMA-ES Optimisation for env_data
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
    print(np.mean(values),i)
    optimizer.tell(solutions, values)

env_data = np.mean(environments, axis=0)

rewards = lab.evaluate_metalearner(meta_learner, number_of_offline_episodes, env_data=scale(env_data, means, stds))
destination_path = sys.argv[3]
file_name = Path(file_path).stem
file = destination_path + "/" + file_name + ".npz"
np.savez(file, online_rewards=online_rewards, generations=generations, environments=environments, rewards=rewards,
         ppo_mean_reward=lab.get_mean_ppo_reward(), true_env=lab.true_env_to_vector())
