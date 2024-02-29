# Implementation of the permutation equivariant Deep Sets network from the
# https://arxiv.org/abs/1703.06114 paper.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import qkeras


class DeepSetsInv(keras.Model):
    """Deep sets permutation invariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        phi_layers: List of number of nodes for each layer of the phi network.
        rho_layers: List of number of nodes for each layer of the rho network.
        activ: String that specifies Activation function to use between the dense layers.
        aggreg: String that specifies the type of aggregator to use after the phi net.
        output_dim: The output dimension of the network. For a supervised task, this is
            equal to the number of classes.
    """
    def __init__(
        self,
        phi_layers: list = [32, 32, 32],
        rho_layers: list = [16],
        output_dim: int = 5,
        activ: str = "relu",
        aggreg: str =  "mean"
    ):
        super(DeepSetsInv, self).__init__(name="InvariantDeepsets")
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers
        self.aggreg = aggreg
        self.activ = activ

        self.build_phi()
        self.build_agg()
        self.build_rho()
        self.output_layer = KL.Dense(self.output_dim, name="OutputLayer")

    def build_phi(self):
        self.phi = keras.Sequential(name="PhiNetwork")
        for layer in self.phi_layers:
            self.phi.add(KL.Dense(layer))
            self.phi.add(KL.Activation(self.activ))

    def build_agg(self):
        switcher = {
            "mean": lambda: tf.reduce_mean,
            "max": lambda: tf.reduce_max,
        }
        self.agg = switcher.get(self.aggreg, lambda: None)()
        if self.agg is None:
            raise ValueError(
                "Given aggregation string is not implemented. "
                "See deepsets.py and add string and corresponding object there."
            )

    def build_rho(self):
        self.rho = keras.Sequential(name="RhoNetwork")
        for layer in self.rho_layers:
            self.rho.add(KL.Dense(layer))
            self.rho.add(KL.Activation(self.activ))

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        agg_output = self.agg(phi_output, axis=1)
        rho_output = self.rho(agg_output)
        logits = self.output_layer(rho_output)

        return logits
