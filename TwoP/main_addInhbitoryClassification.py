# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:17:00 2024

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
# csvDir = 'D:\\fitting_all.csv'
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
errorList = []
for di in range(len(database)):
    # Goes through the pandas dataframe called database created above and
    # if True in the column "Process", the processing continues.

    if database.loc[di]["Process"]:
        try:
            error = False
            dias = []
            xys = []
            entry = database.loc[di]
            baseDir = os.path.join(preprocessDir, entry["Name"], entry["Date"])
            s2pDir = os.path.join(baseDir, "suite2p")
            planeDirs = glob.glob(os.path.join(s2pDir, "plane*",))
            ops = np.load(os.path.join(
                planeDirs[0], "ops.npy"), allow_pickle=True).item()
            ignorePlane = ops["ignore_flyback"]

            # get plane numbers
            planeNums = np.zeros(len(planeDirs))
            for i in range(len(planeDirs)):
                num = re.findall('plane([0-9*]*)', planeDirs[i])
                planeNums[i] = int(num[0])

            keep = ~np.isin(planeNums, ignorePlane)
            planeDirs = np.array(planeDirs)[keep]

            inhType = []
            for i in range(len(planeDirs)):
                currPlane = planeDirs[i]
                matFile = os.path.join(currPlane, "cellTypes.mat")
                if (os.path.exists(matFile)):
                    cellClass = sp.io.loadmat(matFile)
                    inhType.append(cellClass["cClasses"])

            inhType = np.vstack(inhType)

            np.save(os.path.join(baseDir, "rois.isInh"), inhType)
        except:
            print(f"error with {database.loc[i]}")
            print(traceback.format_exc())
            errorList.append(traceback.format_exc())
