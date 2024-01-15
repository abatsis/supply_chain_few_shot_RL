import glob
import os
import re
from copy import deepcopy

import numpy as np

import sys
import pandas as pd
from tqdm import tqdm


def match_input_output(input_array, output, number_of_rows):
    input_array = input_array[0:-1, :]
    output = output[1:, :]
    number_of_rows = number_of_rows - 1
    return input_array, output, number_of_rows


def get_number_of_levels(env_config):
    return len(env_config['r'])


def get_max_lead_time(env_config):
    return max(env_config['L'])


def convert_input_to_array(input):
    if isinstance(input, int):
        input = np.array([input])
    elif isinstance(input, bool):
        input = np.array([float[input]])
    elif isinstance(input, dict):
        input = np.array([10, 100])
    elif isinstance(input, float):
        input = np.array([input])
    return input


class NNDatabase:

    def __init__(self, _input_sizes, _output_size, max_number_of_levels):
        self._output_size = _output_size
        self._output = np.empty([0, _output_size])
        self._input_sizes = _input_sizes
        self._inputs = []
        self._max_number_of_levels = max_number_of_levels
        for size in _input_sizes:
            self._inputs.append(np.empty([0, size]))

    def get_state_features(self, input_array, max_number_of_levels, number_of_levels, max_lead_time):
        bin_size = max_number_of_levels - 1
        required_length = bin_size * (max_lead_time + 1)
        number_of_rows = input_array.shape[0]
        result = np.zeros((number_of_rows, max(required_length, self._input_sizes[0])))

        for i, elem in enumerate(input_array.T):
            bin_number, index = divmod(i, number_of_levels - 1)
            if bin_number == 0:
                result[:, index] = elem
            else:
                bin_number = max_lead_time - bin_number + 1
                result[:, bin_size * bin_number + index] = elem
        return result

    def read_data(self, files):
        # check if directory was given
        if isinstance(files, str) and os.path.isdir(files):
            files = [files + "/" + file for file in os.listdir(files)]

        # initialise temp data holders
        new_inputs = []
        for input in self._inputs:
            new_inputs.append([input])
        new_output = [self._output]

        # read data
        for file in tqdm(files):
            npz_file = re.match(r'.*\.npz$', file)
            if not npz_file:
                continue

            inputs, output = self.read_data_from_file(file)

            new_output.append(output)
            for i, input in enumerate(inputs):
                new_inputs[i].append(input)

        # concatenate and update data
        for i in range(len(self._inputs)):
            self._inputs[i] = np.concatenate(new_inputs[i])
        self._output = np.concatenate(new_output)

        # TODO this a quick workaround
        self._inputs = np.concatenate(self._inputs, axis=1)

    def read_data_from_file(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        files = data.files
        inputs = []
        item = files[-1]
        environment = data[item].item()

        item = files[0]
        input_array = data[item]
        number_of_rows = input_array.shape[0]
        number_of_levels = get_number_of_levels(environment)
        max_lead_time = get_max_lead_time(environment)
        input_array = self.get_state_features(input_array, self._max_number_of_levels, number_of_levels, max_lead_time)

        item = files[1]
        output = data[item]
        number_of_columns = output.shape[1]
        output = np.pad(output, pad_width=((0, 0), (0, self._output_size - number_of_columns)), mode='constant',
                        constant_values=0)

        #input_array, output, number_of_rows = match_input_output(input_array, output, number_of_rows)

        inputs.append(input_array)

        for key, input_size in zip(sorted(environment.keys()), self._input_sizes[-13:]):
            input_array = environment[key]
            input_array = convert_input_to_array(input_array)
            length = len(input_array)
            input_array = np.pad(input_array, pad_width=(0, input_size - length), mode='constant', constant_values=0)
            input_array = np.tile(input_array, (number_of_rows, 1))
            inputs.append(input_array)

        return inputs, output

    def get_inputs(self):
        return self._inputs

    def get_input_sizes(self):
        return self._input_sizes

    def get_output(self):
        return self._output

    def get_output_size(self):
        return self._output_size
