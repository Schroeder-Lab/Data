# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:37:13 2022

@author: LABadmin
"""

import numpy as np
import glob
import os

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:37:13 2022

@author: LABadmin
"""


def get_file_in_directory(directory, simpleName):
    """
    Gets the file path of the first file with the same name.
    For example,if a directory contains two ArduinoInput files, such as
    ArduinoInput0.csv and ArduinoInput1.csv, it will return the file path of
    ArduinoInput0.csv.

    Parameters
    ----------
    directory : str
        The directory where the respective files are located.
    simpleName : str
        a keyword that describes the file. For example, for the arduino input
        file mentioned above, such a keyword would be "ArduinoInput".

    Returns
    -------
    file[0] : str
       The file path of the first file with the same name. .

    """
    file = glob.glob(os.path.join(directory, simpleName + "*"), recursive=True)
    if len(file) > 0:
        return file[0]
    else:
        return None


def get_ops_file(suite2pDir):
    """
    Loads the ops file from the first plane folder in the suite2p folder. Ops file
    is generated directly from suite2p.

    Parameters
    ----------
    suite2pDir : str
        The main directory where the suite2p folders are located.

    Returns
    -------
    ops : dict
        The suite2p ops dictionary.

    """
    # get non backup planes
    combinedDir = glob.glob(os.path.join(
        suite2pDir, "plane*"))
    ops = np.load(
        os.path.join(combinedDir[0], "ops.npy"), allow_pickle=True
    ).item()

    return ops


def _moffat(r, B, A, alpha, beta):
    return B + A * (1 + (((r) ** 2) / alpha**2)) ** -beta


def _gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _linear(x, a, b):
    return a + b * x


def linear_analytical_solution(x, y, noIntercept=False):
    """
    Fits a robust line to data by using the least squares function.

    Parameters
    ----------
    x : np.ndarray
        The values along the x axis.
    y : np.ndarray
        The values along the y axis.
    noIntercept : bool, optional
        Whether to not use the intercept. The default is False (intercept used).

    Returns
    -------
    a : float64
        The intercept of the fitted line.
    b : float64
        The slope of the fitted line.
    mse : float64
        The mean square error of the fit.

    """
    n = len(x)
    a = (np.sum(y) * np.sum(x**2) - np.sum(x) * np.sum(x * y)) / (
        n * np.sum(x**2) - np.sum(x) ** 2
    )
    b = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
        n * np.sum(x**2) - np.sum(x) ** 2
    )
    if noIntercept:
        b = np.sum(x * y) / np.sum(x**2)
    mse = (np.sum((y - (a + b * x)) ** 2)) / n
    return a, b, mse
