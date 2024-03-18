import numpy as np
from tensorflow import keras


class MetaLearner:

    def __init__(self, input_sizes, output_size):
        self.normalisation_layer = keras.layers.Normalization()
        size = sum(input_sizes)
        input = keras.layers.Input(shape=[size])
        normalised_input = self.normalisation_layer(input)
        #normalised_input = keras.layers.GaussianNoise(0.1)(normalised_input)
        # layer = keras.layers.Dropout(0.3)(layer)

        layers = []
        self.ind_models = []

        for _ in range(10):
            layer = keras.layers.Dense(256, activation="elu")(normalised_input)
            #layer = keras.layers.Dropout(0.1)(layer)
            mix = keras.layers.concatenate([normalised_input,layer])
            layer = keras.layers.Dense(256, activation="elu")(mix)
            layer = keras.layers.Dense(output_size, activation="linear")(layer)
            #layer = keras.layers.Dropout(0.1)(layer)
            layers.append(layer)
            ind_model = keras.Model(inputs=input, outputs=layer)
            self.ind_models.append(ind_model)
        layer = keras.layers.average(layers)
        #layer = keras.layers.Dropout(0.1)(layer)
        #layer = keras.layers.Dense(output_size, activation="linear")(layer)
        self.model = keras.Model(inputs=input, outputs=layer)

    def compile(self, loss, optimizer):
        self.model.compile(loss=loss, optimizer=optimizer)
        for model in self.ind_models:
            model.compile(loss=loss, optimizer=optimizer)

    def fit(self, x, y, epochs, validation_data=None, batch_size=None):
        print(x.shape, y.shape)
        self.normalisation_layer.adapt(x, batch_size = 16000)
        self.save_normalisation_data()
        print("finished adaptation")
        for model in self.ind_models:
            model.fit(x=x, y=y, epochs=epochs,
                           validation_data=validation_data,
                           batch_size=batch_size)


    def save(self, file):
        self.model.save(file)

    def set_model(self, model):
        self.model = model

    def predict(self, x):
        ## forward pass instead of predict
        return self.model(x)

    def save_normalisation_data(self):
        mean = self.normalisation_layer.mean.numpy()
        variance = self.normalisation_layer.variance.numpy()
        file = './data/normalisation_data.npz'
        np.savez(file, mean=mean, variance=variance)
