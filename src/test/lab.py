import copy
import os

import numpy as np
import or_gym

env_name = 'InvManagement-v1'


def get_normalisation_data(env_size):
    data = np.load('./data/normalisation_data.npz', allow_pickle=True)
    items = data.files
    mean = data[items[0]][0]
    variance = data[items[1]][0]
    std = np.array([x ** 0.5 for x in variance])
    return mean[-env_size:], std[-env_size:]


def save_explainability_data(observations, actions, env_config, rewards, dir_path, name):
    os.makedirs(dir_path, exist_ok=True)
    file = dir_path + "/" + name + ".npz"
    np.savez(file, obs=observations, actions=actions, rewards=rewards, config=env_config)


class Lab:
    def __init__(self, input_sizes, output_size, max_number_of_levels):
        self._number_of_units = None
        self._env_data = None
        self._env_config = None
        self._input_sizes = input_sizes
        self._output_size = output_size
        self._env_data_size = None
        self._ppo_mean_reward = None
        self._max_number_of_levels = max_number_of_levels

    def convert_input_to_array(self, input):
        if isinstance(input, int):
            input = np.array([input])
        elif isinstance(input, bool):
            input = np.array([float[input]])
        elif isinstance(input, dict):
            input = np.array([input['low'], input['high']])
        elif isinstance(input, float):
            input = np.array([input])
        return input

    def get_state_features(self, input_array):
        bin_size = self._max_number_of_levels - 1
        max_lead_time = max(self._env_config['L'])
        number_of_levels = len(self._env_config['r'])
        required_length = bin_size * (max_lead_time + 1)
        result = np.zeros(max(required_length, self._input_sizes[0]))

        for i, elem in enumerate(input_array):
            bin_number, index = divmod(i, number_of_levels - 1)
            if bin_number == 0:
                result[index] = elem
            else:
                bin_number = max_lead_time - bin_number + 1
                result[bin_size * bin_number + index] = elem
        return result

    def get_data_point(self, obs, env_data):
        obs = self.get_state_features(obs)
        data_point = [obs]
        data_point.extend(env_data)
        data_point = [array.reshape(1, len(array)) for array in data_point]
        # TODO quick workaround
        data_point = np.concatenate(data_point, axis=1)
        return data_point

    def get_action(self, nn_output):
        return nn_output[0, 0:self._number_of_units]

    def env_to_array(self, env_config, input_sizes):
        env_data = []
        for key, input_size in zip(sorted(env_config.keys()), input_sizes[-13:]):
            input_array = env_config[key]
            input_array = self.convert_input_to_array(input_array)
            length = len(input_array)
            input_array = np.pad(input_array, pad_width=(0, input_size - length), mode='constant', constant_values=0)
            env_data.append(input_array)
        return env_data

    def evaluate_metalearner(self, model, num_episodes, env_data=None, logging=False):
        if env_data is None:
            env_data = self._env_data
        else:
            env_data = self.tidy_env_data(env_data)

        env = or_gym.make(env_name, env_config=self._env_config, verbose=0)
        observations = []
        actions = []
        rewards = []
        for i in range(num_episodes):
            obs = env.reset()
            reward = 0.0
            done = False
            while not done:
                data_point = self.get_data_point(obs, env_data)
                nn_output = model.predict(data_point)
                action = self.get_action(nn_output)
                obs, r, done, _ = env.step(action)
                reward += r
                if logging:
                    observations.append(obs)
                    actions.append(action)
            rewards.append(reward)

        if num_episodes == 1:
            return rewards[0]

        return rewards, np.array(observations), np.array(actions)

    def read_environment(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        files = data.files

        item = files[-1]
        self._env_config = data[item].item()
        self._env_data = self.env_to_array(self._env_config, self._input_sizes)
        self._number_of_units = len(self._env_config['I0'])
        self._env_data_size = sum([len(element) for element in self._env_data])

        item = files[2]
        ppo_rewards = data[item]
        self._ppo_mean_reward = np.mean(ppo_rewards)

    def get_env_data_size(self):
        return self._env_data_size

    def get_mean_ppo_reward(self):
        return self._ppo_mean_reward

    def tidy_env_data(self, env_data):
        result = copy.deepcopy(self._env_data)
        index = 0
        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i][j] = env_data[index]
                index += 1
        return result

    def true_env_to_vector(self):
        return np.concatenate(self._env_data)

    def env_config(self):
        return self._env_config
