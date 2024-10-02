# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:22:06 2024

@author: liad0
"""
import glob
import re
import pandas as pd
import scipy as sp
import os
import numpy as np
from Data.TwoP.general import get_file_in_directory


def get_stimulus_info(filePath, props=None):
    """


    Parameters
    ----------
    filePath : str
        the path of the log file.
    props : np.array of str
        the names of the properties to extract, if None looks for a file.
        The default is None.

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the
        props and their values.

    """
    # Gets the experimental details from the props file
    if props is None:
        dirs = glob.glob(os.path.join(filePath, "props*.csv"))
        if len(dirs) == 0:
            print("ERROR: no props file given")
            return None

        props = np.loadtxt(dirs[0], delimiter=",", dtype=str)
        props = props[1:]
    props = np.atleast_1d(props)
    # Gets the log file which contains all the parameters for each stimulus
    # presentation.
    logPath = glob.glob(os.path.join(filePath, "Log*"))
    if len(logPath) == 0:
        return None
    logPath = logPath[
        0
    ]  # Gets the first log file in case there's more than 1.
    # Creates a dictionary for the stimulus properties.
    StimProperties = {}
    # for p in range(len(props)):
    #     StimProperties[props[p]] = []

    searchTerm = ""
    # Finds the different parameters defined in the props file.
    for p in range(len(props)):
        searchTerm += props[p] + "=([a-zA-Z0-9_.-]*)"

        if p < len(props) - 1:
            searchTerm += "|"
    # Reads the log csv file.
    with open(logPath, newline="") as csvfile:
        allLog = csvfile.read()
    # Gets the values for each stimulus repetition for all the parameters.
    for p in range(len(props)):
        m = re.findall(props[p] + "=([.-a-zA-Z0-9_\\\\]*)", allLog)
        # Appends the list of each parameter into a dictionary.
        if len(m) > 0:
            StimProperties[props[p]] = m
    #         # #         stimProps[props[p]] = a[p]
    #         # #     StimProperties.append(stimProps)

    # with open(logPath, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in reader:
    #         # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
    #         m = re.findall(searchTerm, str(row))
    #         if (len(m)>0):
    #             StimProperties.append(m)
    #         # a = []
    #         # for p in range(len(props)):
    #         #     # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
    #         #     m = re.findall(props[p]+'=([a-zA-Z0-9_.-]*)', row[np.min([len(row)-1,p])])
    #         #     if (len(m)>0):
    #         #         # a.append(m[0])
    #         #         StimProperties[props[p]].append(m[0])
    #         # # if (len(a)>0):
    #         # #     stimProps = {}
    #         # #     for p in range(len(props)):
    #         # #         stimProps[props[p]] = a[p]
    #         # #     StimProperties.append(stimProps)
    # Returns a pandas dataframe of the stimulus properties dictionary.
    return pd.DataFrame(StimProperties)


def get_sparse_noise(filePath, size=None):
    """
    Pulls the sparse noise from the directory.

    Parameters
    ----------
    filePath : str
        The full file path for the sparse noise file.
    size: tuple
        A tuple for the size of the screen (into how many squares the screen
        is divided into). The default is None.

    Returns
    -------
    np.array [frames X size[0] X size[1]]
        The sparse map.
    """

    # Loads sparse noise binary file.
    filePath_ = get_file_in_directory(filePath, "sparse")
    sparse = np.fromfile(filePath_, dtype=np.dtype("b")).astype(float)

    if size is None:
        # Gets experimental details (size of the screen) from the props file.
        dirs = glob.glob(os.path.join(filePath, "props*.csv"))
        if len(dirs) == 0:
            print("ERROR: no props file given")
            return None
        # Gets the size of the squares from the props.
        size = np.loadtxt(dirs[0], delimiter=",", dtype=str)
        size = size[1:].astype(int)
    # Reassigns values in the sparse array.
    sparse[sparse == -128] = 0.5
    sparse[sparse == -1] = 1
    # Reshapes the sparse array to represent the size of the screen and where
    # within this grid the black or white squares appeared.
    sparse = np.reshape(
        sparse, (int(len(sparse) / (size[1] * size[0])), size[0], size[1])
    )
    # Rearranges the sparse map.
    return np.moveaxis(np.flip(sparse, 2), -1, 1)
