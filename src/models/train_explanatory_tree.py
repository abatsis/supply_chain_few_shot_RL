import argparse
import os
import random
import math
from NNDatabase import NNDatabase
from feature_name_generation import get_features
from sklearn.tree import DecisionTreeRegressor, plot_tree
import dtreeviz


def main():
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
    for max_leaf_nodes in [10]:
        tree_regressor = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=random_state)
        print(train_input.shape, train_output.shape, type(train_input))
        tree_regressor.fit(train_input, train_output)
        score = tree_regressor.score(val_input, val_output)
        result = Result(tree_regressor, score, max_leaf_nodes)
        results.append(result)
        print(f'max_depth: {max_leaf_nodes}, score: {score}')

    model = get_best_result(results).model
    features = get_features(input_sizes, max_number_of_levels)
    viz_model = dtreeviz.model(model=model,
                               X_train=train_input,
                               y_train=train_output,
                               feature_names=features,
                               target_name='action')

    save_plot(viz_model)


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


def save_plot(viz_model):
    plot = viz_model.view()
    destination_path = 'reports/explainability/'
    os.makedirs(destination_path, exist_ok=True)
    file_name = f'treeviz_{data_type}.svg' if not hidden_context else f'treeviz_hidden_{data_type}.svg'
    plot.save(destination_path + file_name)


if __name__ == "__main__":
    main()
