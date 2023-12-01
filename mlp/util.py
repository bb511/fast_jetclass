# Utility methods for the mlp network training, testing, etc...

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity

from mlp.mlp import MLP
from mlp.mlp import MLPRegular
from mlp.mlp_quantised import MLPRegularQuantised
from mlp.mlp_quantised import mlp_regularised_synth
from util.terminal_colors import tcols


def choose_mlp(
    mlp_type: str,
    train_njets: int,
    nconst: int,
    nfeats: int,
    model_hyperparams: dict,
    compilation_hyperparams: dict,
    training_hyperparams: dict,
) -> keras.models.Model:
    """Select and instantiate a certain type of MLP."""
    print("Instantiating model with the hyperparameters:")
    for key in model_hyperparams:
        print(f"{key}: {model_hyperparams[key]}")


    mlp_type, model_hyperparams = check_quantised_model(mlp_type, model_hyperparams)
    model_hyperparams = check_synthesis_model(mlp_type, model_hyperparams)

    switcher = {
        "mlp":      lambda: MLP(**model_hyperparams),
        "mlp_reg":  lambda: MLPRegular(**model_hyperparams),
        "qmlp_reg": lambda: MLPRegularQuantised(**model_hyperparams)
        "qsmlp_reg": lambda: mlp_regularised_synth(**model_hyperparams),
    }

    model = switcher.get(mlp_type, lambda: None)()

    if 'pruning_rate' in training_hyperparams:
        if training_hyperparams["pruning_rate"] > 0:
            nsteps = train_njets // training_hyperparams["batch"]
            model = prune_model(model, nsteps, training_hyperparams["pruning_rate"])

    comp_hps = {}
    comp_hps.update(compilation_hyperparams)
    comp_hps["optimizer"] = load_optimizer(
        comp_hps["optimizer"], training_hyperparams["lr"]
    )
    comp_hps["loss"] = choose_loss(compilation_hyperparams["loss"])

    model.compile(**comp_hps)
    model.build((None, nconst, nfeats))
    print(tcols.OKGREEN + "Model compiled and built!" + tcols.ENDC)

    return model


def load_optimizer(choice: str, lr: float) -> keras.optimizers.Optimizer:
    """Construct a keras optimiser object with a certain learning rate."""

    switcher = {
        "adam": lambda: keras.optimizers.Adam(learning_rate=lr),
    }

    optimiser = switcher.get(choice, lambda: None)()

    return optimiser


def choose_loss(choice: str, from_logits: bool = True) -> keras.losses.Loss:
    """Construct a keras optimiser object with a certain learning rate."""

    switcher = {
        "categorical_crossentropy": lambda: keras.losses.CategoricalCrossentropy(),
        "softmax_with_crossentropy": lambda: tf.nn.softmax_cross_entropy_with_logits,
    }

    loss = switcher.get(choice, lambda: None)()

    return loss


def print_training_attributes(model: keras.models.Model, args: dict):
    """Prints model attributes so all interesting infromation is printed."""
    compilation_hyperparams = args["compilation"]
    train_hyperparams = args["training_hyperparams"]

    print("\nTraining parameters")
    print("-------------------")
    print(tcols.OKGREEN + "Optimiser: \t" + tcols.ENDC, model.optimizer.get_config())
    print(tcols.OKGREEN + "Batch size: \t" + tcols.ENDC, train_hyperparams["batch"])
    print(tcols.OKGREEN + "Learning rate: \t" + tcols.ENDC, train_hyperparams["lr"])
    print(tcols.OKGREEN + "Training epochs:" + tcols.ENDC, train_hyperparams["epochs"])
    print(tcols.OKGREEN + "Loss: \t\t" + tcols.ENDC, compilation_hyperparams["loss"])
    print("")


def prune_model(model, nsteps: int, pruning_rate: float = 0.5):
    """Prune the weights of a model during training."""
    def prune_function(layer):
        pruning_params = {
            "pruning_schedule": sparsity.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=pruning_rate,
                begin_step=nsteps * 2,
                end_step=nsteps * 10,
                frequency=nsteps,
            )
        }
        if isinstance(layer, tf.keras.layers.Dense) and layer.name != "output":
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    model = tf.keras.models.clone_model(model, clone_function=prune_function)

    return model


def check_quantised_model(model_type: str, model_hyperparams: dict):
    """Check if one should impose any quantisation on the model."""
    if 'nbits' in model_hyperparams:
        if model_hyperparams["nbits"] > 0:
            model_type = "q" + mlp_type
        else:
            model_hyperparams.pop("nbits")

    return model_type, model_hyperparams


def check_synthesis_model(model_type: str, model_hyperparams: dict):
    """Check if the model is meant to be synthesized (uses different implementation)."""
    if model_type[0] == "s" or model_type[1] == "s":
        model_hyperparams.update({"input_shape": (nconst, nfeats)})

    return model_hyperparams
