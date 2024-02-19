# Feature selection methods partaining for the jet particle data.

import h5py
import numpy as np


def get_features_numpy(data: np.ndarray, feat_selection: str):
    """Choose what feature selection to employ on the data. Return shape."""
    switcher = {
        "ptetaphi": lambda: select_features_ptetaphi_numpy(data),
        "allfeats": lambda: select_features_all_numpy(data),
    }

    data = switcher.get(feat_selection, lambda: None)()
    if data is None:
        raise TypeError("Feature selection name not valid!")

    return data


def select_features_ptetaphi_numpy(data: np.ndarray):
    """Selects (pT, etarel, phirel) features from the numpy jet array."""
    return data[:, :, [5, 8, 11]]


def select_features_all_numpy(data: np.ndarray):
    """Gets all the features from the numpy jet array.

    The features in this kind of 'selection' are:'
    (px, py, pz, E, Erel, pT, ptrel, eta, etarel, etarot, phi, phirel, phirot, deltaR,
    cos(theta), cos(thetarel), pdgid)
    """
    return data[:, :, :]
