# Deepsets models in a format friendly for synthetisation. For more details on the
# architecture see the deepsets.py file.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import qkeras


def mlp_regularised_synth(
    input_shape: tuple,
    layers: list,
    activ: str = "relu",
    nbits: int = 8,
    **kwargs
):
    """Quantised MLP like in mlp_quantised.py but in a format consistent with hls4ml."""

    nclasses = 5
    nbits = format_quantiser(nbits)
    activ = format_qactivation(activ, nbits)

    mlp_input = keras.Input(shape=input_shape, name="input_layer")
    mlp_layer = KL.QDense(
        layers[0],
        kernel_regularizer=keras.regularizers.L1(kwargs['l1_coeff']),
        bias_quantizer=nbits,
        kernel_quantizer=nbits,
    )(mlp_layer)

    for layer_nodes in layers.pop(0):
        if 'l1_coeff' in kwargs:
            mlp_layer = KL.QDense(
                layer_nodes,
                kernel_regularizer=keras.regularizers.L1(kwargs['l1_coeff']),
                bias_quantizer=nbits,
                kernel_quantizer=nbits,
            )(mlp_layer)
        else:
            mlp_layer = KL.QDense(
                layer_nodes,
                bias_quantizer=nbits,
                kernel_quantizer=nbits,
            )(mlp_layer)
        mlp_activ = KL.Activation(activ)(mlp_layer)

    mlp_layer = KL.Dense(nclasses)
    mlp_output = KL.Softmax()(mlp_layer)
    deepsets = keras.Model(mlp_input, deepsets_output, name="deepsets_invariant")

    return deepsets


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
