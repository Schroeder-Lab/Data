import numpy as np
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl
import pandas as pd
import os
import glob
import pickle
import traceback

from Data.TwoP.general import get_ops_file
from Data.TwoP.runners import *
from Data.Bonsai.extract_data import *
from Data.user_defs import *
from Data.Bonsai.behaviour_protocol_functions import stimulus_prcoessing_dictionary

# TODO: change folder_defs to user_defs and move one level up

# %% load directories and processing ops

# Please change the values in define_directories and create_processing_ops in
# module folder_defs.
dirs = define_directories()

csvDir = dirs["dataDefFile"]
s2pDir = dirs["preprocessedDataDir"]
zstackDir = dirs["zstackDir"]
metadataDir = dirs["metadataDir"]
pops = create_2p_processing_ops()

# %% read database

# In the file the values should be Name, Date, Zstack dir number, planes to
# ignore and save directory (if none default is wanted) and process (True,False)
database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Zstack": str,
        "IgnorePlanes": str,
        "SaveDir": str,
        "Process": bool,
    },
)

# %% run over data base
for i in range(len(database)):
    # Goes through the pandas dataframe called database created above and
    # if True in the column "Process", the processing continues.
    if database.loc[i]["Process"]:
        try:
            print("reading directories" + str(database.loc[i]))
            (
                s2pDirectory,
                zstackPath,
                metadataDirectory,
                saveDirectory,
            ) = read_csv_produce_directories(
                database.loc[i], s2pDir, zstackDir, metadataDir
            )
            # Converts and places the planes to be ignored in an array that
            # is at least 1-dimensional.
            ignorePlanes = np.atleast_1d(
                np.array(database.loc[i]["IgnorePlanes"]).astype(int)
            )
            # make save directory the main root
            saveDirectory = os.path.split(s2pDirectory)[0]
            # Returns the ops dictionary.
            ops = get_ops_file(s2pDirectory)
            if pops["process_suite2p"]:

                print("getting piezo data")
                # Returns the movement of the piezo within one frame across the
                # z-axis for all planes.
                planePiezo = get_piezo_data(ops)
                print("processing suite2p data")
                try:
                    fc = process_s2p_directory(
                        s2pDirectory,
                        pops,
                        planePiezo,
                        zstackPath,
                        saveDirectory=saveDirectory,
                        ignorePlanes=ignorePlanes,
                        debug=pops["debug"],
                    )
                except Exception:
                    print("Could not process due to errors, moving to next batch.")
                    print(traceback.format_exc())
            if pops["process_bonsai"]:
                print("reading bonsai data")
                process_metadata_directory(
                    metadataDirectory, ops, pops, saveDirectory
                )

        except Exception:
            print("Could not process due to errors, moving to next batch.")
            print(traceback.format_exc())
        plt.close('all')
    # if False in the column "Process", the processing of those experiments is
    # skipped.
    else:
        print("skipping " + str(database.loc[i]))

    # %%
