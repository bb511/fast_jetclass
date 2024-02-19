# Quantised deepsets networks equivalent to the float32 implementations in deepsets.py.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import qkeras


class MLPRegularQuantised(keras.Model):
    """Quantised weights MLP with regularisation..

    Attributes:
        layers: List, where each element represents the number of nodes in a certain
            layer. The lenght of the list is equal to the depth of the network.
        activ: Activation function to use between the dense layers.
        kwargs: Regularisation parameters.
    """

    def __init__(self, layers: list, activ: str = "relu", nbits: int = 8, **kwargs):
        super(MLPRegularQuantised, self).__init__(name="MLPRegularQuantised")
        self.nclasses = 5

        quantizer = format_quantiser(nbits)
        activ = format_qactivation(activ, nbits)

        self.mlp = keras.Sequential()
        for layer_nodes in layers:
            if "l1_coeff" in kwargs:
                self.mlp.add(
                    qkeras.QDense(
                        layer_nodes,
                        kernel_regularizer=keras.regularizers.L1(kwargs["l1_coeff"]),
                        bias_quantizer=quantizer,
                        kernel_quantizer=quantizer,
                    )
                )
            else:
                self.mlp.add(
                    qkeras.QDense(
                        layer_nodes,
                        bias_quantizer=quantizer,
                        kernel_quantizer=quantizer,
                    )
                )
            self.mlp.add(qkeras.QActivation(activ))

        self.mlp.add(KL.Dense(self.nclasses))

    def call(self, inputs: np.ndarray, **kwargs):
        inputs = KL.Flatten()(inputs)
        return self.mlp(inputs)


def format_quantiser(nbits: int):
    """Format the quantisation of the ml floats in a QKeras way."""
    if nbits == 1:
        return "binary(alpha=1)"
    elif nbits == 2:
        return "ternary(alpha=1)"
    else:
        return f"quantized_bits({nbits}, 0, alpha=1)"


def format_qactivation(activation: str, nbits: int) -> str:
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({nbits}, 0)"
