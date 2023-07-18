from pathlib import Path

import numpy as np


class Normaliser:
    def __init__(self, standardisation_data=None, file_name=None):
        if file_name is None:
            pass
        else:
            type = 'test' if 'test' in file_name else 'train'
            file_name = Path(file_name).stem
            path = f'./data/{type}/{file_name}.npz'
            data = np.load(path, allow_pickle=True)
            files = data.files
            standardisation_data = data[files[2]]

        standardisation_data = [abs(x) for x in standardisation_data]
        self._mean = np.mean(standardisation_data)

    def __call__(self, a_list):
        return [x / self._mean for x in a_list]

    def mean(self):
        return self._mean
