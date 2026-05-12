# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:37:13 2022

@author: LABadmin
"""
import sys

import numpy as np
import glob
import os

import pandas as pd

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


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def make_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    tee = Tee(sys.__stdout__, log_file)
    return log_file, tee


def make_incremental_log_path(log_dir: str, base_name: str = "main_metadata", ext: str = ".log.txt") -> str:
    os.makedirs(log_dir, exist_ok=True)
    i = 1
    while True:
        candidate = os.path.join(log_dir, f"{base_name}_{i:04d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def make_paths(dataset: pd.Series, directories: dict) -> dict:
    """
    Build paths for a single experiment row.

    Parameters
    ----------
    dataset : pandas Series
        A single row (experiment) from the preprocess DataFrame.
    directories : dict
        Base directories dictionary.

    Returns
    -------
    dict
        Paths for tiffs, suite2p, piezo, and output.
    """
    subject = str(dataset.Name)
    date = str(dataset.Date)
    paths = {
        "raw": directories["raw"].format(Name=subject, Date=date),
        "suite2p": directories["suite2p"].format(Name=subject, Date=date),
        "output": directories["output"].format(Name=subject, Date=date),
    }

    if not os.path.exists(paths["suite2p"]):
        paths["suite2p"] = None
        print(
            f"NOTE: Suite2p directory for {dataset['Name']} {dataset['Date']} not found. Skipping.\n\n"
        )
        return paths

    os.makedirs(paths["output"], exist_ok=True)
    return paths
