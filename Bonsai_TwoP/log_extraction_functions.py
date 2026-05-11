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
from TwoP.general import get_file_in_directory


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
    # Get stimulus variables from props file (if not provided).
    if props is None:
        dirs = glob.glob(os.path.join(filePath, "props*.csv"))
        if len(dirs) == 0:
            print("ERROR: no props file given")
            return None

        props = np.loadtxt(dirs[0], delimiter=",", dtype=str)[1:]

    props = np.atleast_1d(props)

    # Gets recorded data from the log file.
    log_paths = glob.glob(os.path.join(filePath, "Log*"))
    if not log_paths:
        return None

    # Build one regex per prop: prop=<value>
    # Value chars follow your existing pattern.
    patterns = {
        prop: re.compile(rf"{re.escape(prop)}=([.\-a-zA-Z0-9_\\]*)")
        for prop in props
    }

    rows = []
    with open(log_paths[0], "r", newline="") as f:
        for line_no, line in enumerate(f):
            record = {"line": line_no}
            all_found = True

            for prop, pat in patterns.items():
                m = pat.search(line)
                if m is None:
                    all_found = False
                    break
                record[prop] = m.group(1)

            if all_found:
                rows.append(record)

    # Stable column order: line first, then props in input order
    return pd.DataFrame(rows, columns=["line", *props.tolist()])


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
