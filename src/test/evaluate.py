import re
import secrets
import sys
from pathlib import Path
import numpy as np
import or_gym
from tqdm import tqdm
from tensorflow import keras
import pandas as pd
from lab import Lab
from lab import save_explainability_data
sys.path.append('src/models/')
from meta_learner import MetaLearner

input_sizes = [170, 11, 11, 1, 1, 11, 1, 2, 11, 11, 1, 1, 11, 1]
output_size = 11
lab = Lab(input_sizes, output_size)


def evaluate_rl_on_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    files = data.files

    return data[files[2]]


model = keras.models.load_model(sys.argv[1])
meta_learner = MetaLearner(input_sizes, output_size)
meta_learner.set_model(model)
file_path = sys.argv[2]
destination_path = sys.argv[3]
lab.read_environment(file_path)

rewards_nn, observations, actions = lab.evaluate_metalearner(meta_learner, 20, logging=True)
rewards_rl = evaluate_rl_on_file(file_path)
data = {'RL': rewards_rl, 'Metalearner': rewards_nn}
data_frame = pd.DataFrame(data)
file_name = Path(file_path).stem
file = f"{destination_path}/{file_name}.csv"
data_frame.to_csv(file)
save_explainability_data(observations, actions, lab.env_config(), rewards_nn, 'data/explainability', file_name)
