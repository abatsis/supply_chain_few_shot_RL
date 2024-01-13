import math
import os
import random
import numpy as np
from meta_learner import MetaLearner
from NNDatabase import NNDatabase
from datetime import datetime
import tensorflow as tf

import sys
from keras.optimizers.adamw import AdamW
from sklearn.metrics import mean_squared_error as MSE


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

meta_learner = MetaLearner(input_sizes, output_size)
meta_learner.compile(loss="mean_squared_error", optimizer="adam")

inputs = train_database.get_inputs()
output = train_database.get_output()

val_inputs = validation_database.get_inputs()
val_outputs = validation_database.get_output()


print(meta_learner.model.summary())

with tf.device('/CPU:0'):
    inputs = tf.constant(inputs)
    outputs = tf.constant(output)
    val_inputs = tf.constant(val_inputs)
    val_outputs = tf.constant(val_outputs)

history = meta_learner.fit(
    x=inputs,
    y=output,
    epochs=2,
    validation_data=(val_inputs, val_outputs),
    batch_size=1024,
)

val_outputs_hat = meta_learner.model.predict(val_inputs, batch_size=256)
print("MSE", MSE(val_outputs, val_outputs_hat))

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
os.makedirs(model_dir, exist_ok=True)
model_name = (
    sys.argv[1] if len(sys.argv) > 1 else date_time.strftime("%d-%m-%Y_%H:%M:%S")
)
file = f"{model_dir}/{model_name}"
print(file)
meta_learner.save(file)
