# Implementation of the permutation equivariant Deep Sets network from the
# https://arxiv.org/abs/1703.06114 paper.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import qkeras


class MLP(keras.Model):
    """Simple MLP implementation with variable number of layers and nodes per layer.

    Attributes:
        layers: List, where each element represents the number of nodes in a certain
            layer. The lenght of the list is equal to the depth of the network.
        activ: Activation function to use between the dense layers.
    """

    def __init__(self, layers: list, activ: str = "relu", **kwargs):
        super(MLP, self).__init__(name="MLP")
        self.nclasses = 5

        self.mlp = keras.Sequential()
        for layer_nodes in layers:
            self.mlp.add(KL.Dense(layer_nodes))
            self.mlp.add(KL.Activation(activ))

        self.mlp.add(KL.Dense(self.nclasses))

    def call(self, inputs: np.ndarray, **kwargs):
        inputs = KL.Flatten()(inputs)
        return self.mlp(inputs)


class MLPRegular(keras.Model):
    """Same as above, but this time with regularisation, L1 and Dropout.

    Attributes:
        layers: List, where each element represents the number of nodes in a certain
            layer. The lenght of the list is equal to the depth of the network.
        activ: Activation function to use between the dense layers.
        kwargs: Regularisation parameters.
    """

    def __init__(self, layers: list, activ: str = "relu", **kwargs):
        super(MLPRegular, self).__init__(name="MLPRegularised")
        self.nclasses = 5

        self.mlp = keras.Sequential()
        for layer_nodes in layers:
            if 'l1_coeff' in kwargs:
                self.mlp.add(
                    KL.Dense(
                        layer_nodes,
                        kernel_regularizer=keras.regularizers.L1(kwargs['l1_coeff']))
                )
            else:
                self.mlp.add(KL.Dense(layer_nodes))
            if 'dropout_rate' in kwargs:
                self.mlp.add(KL.Dropout(kwargs['dropout_rate']))
            self.mlp.add(KL.Activation(activ))

        self.mlp.add(KL.Dense(self.nclasses))

    def call(self, inputs: np.ndarray, **kwargs):
        inputs = KL.Flatten()(inputs)
        return self.mlp(inputs)
