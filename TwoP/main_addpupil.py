# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:29:37 2024

@author: liad0
"""
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

# %%
dirs = define_directories()
csvDir = dirs["dataDefFile"]
preprocessDir = dirs["preprocessedDataDir"]
# %%
database = pd.read_csv(
    csvDir, usecols=["Name", "Date", "Process"],
    dtype={
        "Name": str,
        "Date": str,
        "Process": bool,
    },
)

# %%
for i in range(len(database)):
    # Goes through the pandas dataframe called database created above and
    # if True in the column "Process", the processing continues.
    if database.loc[i]["Process"]:
        entry = database.loc[i]
        baseDir = os.path.join(preprocessDir, entry["Name"], entry["Date"])
        s2pDir = os.path.join(baseDir, "suite2p")
        pupilDir = os.path.join(baseDir, "pupil", "xyPos_diameter")
        opsFile = glob.glob(os.path.join(s2pDir, "plane[0-9]", "ops.npy"))
        if (len(opsFile) == 0):
            print('No suite2p cannot continue')
            continue
        ops = np.load(opsFile[0], allow_pickle=True).item()
