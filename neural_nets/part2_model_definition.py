from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tensorflow.keras import layers


def build_default_regression_model(num_inputs, num_outputs):
    # Przykładowa prosta sieć neuronowa do zadania regresji

    model = tf.keras.Sequential()
    model.add(layers.Dense(20, activation='relu', input_shape=[num_inputs], name='ukryta_1'))
    model.add(layers.Dense(40, activation='relu', name='ukryta_2'))
    model.add(layers.Dense(30, activation='relu', name='ukryta_3'))
    model.add(layers.Dense(num_outputs, name='wyjsciowa'))  # model regresyjny, activation=None w warstwie wyjściowej

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.mse,
                  metrics=[tf.keras.metrics.mean_absolute_error, 'mse'])

    return model


@dataclass
class FFN_Hyperparams:
    # stale (zalezne od problemu)
    num_inputs: int
    num_outputs: int

    # do znalezienia optymalnych (np. metodą Random Search) [w komentarzu zakresy wartości, w których można szukać]
    hidden_dims: List[int]              # [10] -- [100, 100, 100]
    activation_fcn: str                 # wybor z listy, np: ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'softplus']
    learning_rate: float                # od 0.1 do 0.00001 (losowanie eksponensu 10**x, gdzie x in [-5, -1])


def build_model(hp: FFN_Hyperparams):
    model = tf.keras.Sequential()

    # Loop over hidden_dims list to add hidden layers
    for i, dim in enumerate(hp.hidden_dims):
        # Add hidden layer with specified dimension and activation function
        model.add(layers.Dense(dim, activation=hp.activation_fcn, name=f'ukryta_{i + 1}'))

    # Add output layer with num_outputs
    model.add(layers.Dense(hp.num_outputs, name='wyjsciowa'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.mse,
                  metrics=[tf.keras.metrics.mean_absolute_error, 'mse'])

    return model

