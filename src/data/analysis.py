import os
import sys
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np


def stats_of_array(array):
    stats = [np.mean(array), np.std(array), np.min(array)]
    stats.extend(np.quantile(array, [0.025, 0.975]))
    stats.append(np.max(array))
    return stats


def get_stats(index, size):
    grouped_data = dict()
    for position, value in enumerate(size):
        key = index[position]
        if key in grouped_data:
            grouped_data[key].append(value)
        else:
            grouped_data[key] = [value]
    for key in grouped_data.keys():
        grouped_data[key] = stats_of_array(grouped_data[key])

    stats = [grouped_data[key] for key in sorted(grouped_data.keys())]

    return stats


def convert_input_to_array(input):
    if isinstance(input, int):
        input = np.array([input])
    elif isinstance(input, bool):
        input = np.array([float[input]])
    elif isinstance(input, dict):
        input = np.array([input['low'], input['high']])
    elif isinstance(input, float):
        input = np.array([input])
    return input


def read_data_from_file(file_path):
    index_data = []
    sizes = []
    output_sizes = []

    data = np.load(file_path, allow_pickle=True)
    files = data.files
    inputs = []
    index = 0

    item = files[0]
    input_array = data[item]
    number_of_columns = input_array.shape[1]
    index_data.append(index)
    sizes.append(number_of_columns)
    index += 1

    item = files[1]
    output = data[item]
    number_of_columns = output.shape[1]
    output_sizes.append(number_of_columns)

    item = files[-1]
    environment = data[item].item()
    for key in sorted(environment.keys()):
        input_array = environment[key]
        input_array = convert_input_to_array(input_array)
        length = len(input_array)
        sizes.append(length)
        index_data.append(index)
        index += 1

    return index_data, sizes, output_sizes


def load_data(folder):
    path = "./data/"
    path = path + folder + '/'
    files = [path + file for file in os.listdir(path) if not file.startswith('.')]
    index = []
    size = []
    output_size = []
    for file in files:
        index_data, sizes, output_sizes = read_data_from_file(file)
        index.extend(index_data)
        size.extend(sizes)
        output_size.extend(output_sizes)

    return index, size, output_size


sns.set()
data_type = sys.argv[1]
os.makedirs('reports/figures', exist_ok=True)
index, size, output_size = load_data(data_type)
data = {'index': index, 'size': size}

data_frame = pd.DataFrame(data)
sns.violinplot(data=data_frame, x='index', y="size")
plt.savefig(f"./reports/figures/data_input_sizes_{data_type}.pdf", dpi=300)
plt.clf()

data = {'output_size': output_size}
data_frame = pd.DataFrame(data)
sns.violinplot(data=data_frame, y='output_size')
plt.savefig(f"./reports/figures/data_output_sizes_{data_type}.pdf", dpi=300)
plt.clf()

stats= get_stats(index, size)
columns = ['Mean value', 'Standard', 'Min', 'Quantile 0.025', 'Quantile 0.975', 'Max']
rows = [str(i) for i in range(len(stats))]

data_frame = load_data(data_type)
table_frame = pd.DataFrame(stats, rows, columns)
table_frame_out = pd.DataFrame([stats_of_array(output_size)], columns=columns)

with open(f'./reports/generated_data_report_{data_type}.txt', 'w') as f:
    print(table_frame, file=f)
    print(table_frame.to_latex(), file=f)
    print(table_frame, file=f)
    print(table_frame.to_latex(), file=f)
