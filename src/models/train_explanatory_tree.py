import argparse
import os
import random
import math
from NNDatabase import NNDatabase
from feature_name_generation import get_features
from sklearn.tree import DecisionTreeRegressor, plot_tree
import dtreeviz

input_sizes = [170, 11, 11, 1, 1, 11, 1, 2, 11, 11, 1, 1, 11, 1]
output_size = 11
max_number_of_levels = 9

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', action='store_true')
parser.add_argument('--skewed', action='store_true')
args = parser.parse_args()
hidden_context = vars(args)['hidden']
skewed = vars(args)['skewed']
data_type = 'test_skewed' if skewed else 'test'


def get_data_path():
    if hidden_context:
        data_path = f'data/explainability/{data_type}_hidden_context/'
    else:
        data_path = f'data/explainability/{data_type}/'
    return data_path


class Result:
    def __init__(self, model, score, max_depth):
        self.model = model
        self.score = score
        self.max_depth = max_depth


def get_best_result(results):
    scores = [result.score for result in results]
    index = scores.index(max(scores))
    return results[index]


def get_training_data(train_database, validation_database):
    train_input = train_database.get_inputs()
    train_output = train_database.get_output()[:, 0]
    val_input = validation_database.get_inputs()
    val_output = validation_database.get_output()[:, 0]
    return train_input, train_output, val_input, val_output


path = get_data_path()
files = [path + file for file in os.listdir(path)]
train_set_percentage = math.floor(len(files) * 0.9)

random.shuffle(files)
train_database = NNDatabase(input_sizes, output_size, max_number_of_levels)
validation_database = NNDatabase(input_sizes, output_size, max_number_of_levels)

train_database.read_data(files[0:train_set_percentage])
validation_database.read_data(files[train_set_percentage:])
train_input, train_output, val_input, val_output = get_training_data(train_database, validation_database)

random_state = 1234  # get reproducible trees
results = []
for max_depth in range(1, 5):
    tree_regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state, criterion="absolute_error")
    tree_regressor.fit(train_input, train_output)
    score = tree_regressor.score(val_input, val_output)
    result = Result(tree_regressor, score, max_depth)
    results.append(result)
    print(f'max_depth: {max_depth}, score: {score}')

model = get_best_result(results).model
features = get_features(input_sizes, max_number_of_levels)
viz_model = dtreeviz.model(model=model,
                           X_train=train_input,
                           y_train=train_output,
                           feature_names=features,
                           target_name='action')

plot = viz_model.view()
file_path = f'treeviz_{data_type}.svg' if not hidden_context else f'treeviz_hidden_{data_type}.svg'
plot.save(file_path)
