import math
import os
import random

from NNDatabase import NNDatabase
from datetime import datetime
import sys
from lightgbm.sklearn import LGBMRegressor
from sklearn import metrics
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor

def get_optimal_number_of_epochs(loss_list):
    return loss_list.index(min(loss_list)) + 1


input_sizes = [170, 11, 11, 1, 1, 11, 1, 2, 11, 11, 1, 1, 11, 1]
output_size = 11
max_number_of_levels = 9
path = "./data/train/"
files = [path + file for file in os.listdir(path)]
train_set_percentage = math.floor(len(files) * 0.9)

random.shuffle(files)
train_database = NNDatabase(input_sizes, output_size, max_number_of_levels)
validation_database = NNDatabase(input_sizes, output_size, max_number_of_levels)
train_database.read_data(files[0:train_set_percentage])
validation_database.read_data(files[train_set_percentage:])

inputs = train_database.get_inputs()
output = train_database.get_output()

val_inputs = validation_database.get_inputs()
val_outputs = validation_database.get_output()


# clf = LGBMRegressor(n_jobs=24,
# #                             n_estimators=2000,
# #                             num_leaves=200, linear_tree=True,
# #                             #categorical_features = list(range(X.shape[1])),
#                             verbose =1, is_training_metric = True)


#clf = MultiOutputRegressor(clf, n_jobs=2)
clf = ExtraTreesRegressor(n_estimators=1000, max_depth=200, n_jobs=200)
#clf.fit(inputs, output)
clf.fit(X= inputs,y =  output)
 #       eval_set=[(val_inputs, val_outputs)],
 #        early_stopping_rounds=100)
# n_estimators_ = clf.best_iteration_
# print(f"n_estimators = {n_estimators_}")

mse = metrics.mean_squared_error(val_outputs, clf.predict(val_inputs))

print("mse",mse)



# validation_losses = history["val_loss"]
# optimal_number_of_epochs = get_optimal_number_of_epochs(validation_losses)
#
# # append validation data to train data
# inputs = [
#     np.concatenate([input_array, val_input])
#     for input_array, val_input in zip(inputs, val_inputs)
# ]
# output = np.concatenate([output, val_outputs])
#
# meta_learner = MetaLearner(input_sizes, output_size)
# meta_learner.compile(loss="mean_squared_error", optimizer=AdamW())
# meta_learner.fit(x=inputs, y=output, epochs=optimal_number_of_epochs)

date_time = datetime.now()

model_dir = f"./models"
model_name = (
    sys.argv[1] if len(sys.argv) > 1 else date_time.strftime("%d-%m-%Y_%H:%M:%S")
)
file = f"{model_dir}/{model_name}"
meta_learner.save(file)
