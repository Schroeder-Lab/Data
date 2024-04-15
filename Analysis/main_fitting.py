# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:06:33 2023

@author: Liad
"""
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
import numpy as np
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl
import sys
import dask.array as da
import pandas as pd
import re
import traceback
from numba import jit, cuda

from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

import os
import glob
import pickle
from numba import jit

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import acf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf

# note: probably need to set your path if this gives a ModuleNotFoundError
from alignment_functions import get_calcium_aligned
from support_functions import (
    load_grating_data,
    get_trial_classification_running,
    run_complete_analysis,
    get_directory_from_session,
    remove_blinking_trials
)

from Data.user_defs import define_directories
from plotting_functions import print_fitting_data
from user_defs import directories_to_fit, create_fitting_ops
import traceback
from abc import ABC, abstractmethod
import inspect


# %%
# Note: go to user_defs and change the inputs to directories_to_fit() and create_fitting_ops().


ops = create_fitting_ops()
csvDir = ops["fitting_list"]

sessions = pd.read_csv(csvDir)
# sessions.insert(2, column='SpecificNeurons', value=[
#                 [] for _ in range(len(sessions))])
sessions = sessions[['Name', 'Date',
                     'SpecificNeurons', 'Process']].to_dict('records')

# Loads the save directory from the fitting_ops in user_defs.
saveDirBase = ops["save_dir"]
processedDataDir = ops["processed files"]
for currSession in sessions:
    if (not currSession['Process']):
        print(f"Process set to False in session: {currSession}")
        continue
    plt.close('all')
    print(f"starting to run session: {currSession}")
    # Gets data for current session from the Preprocessed folder.
    try:
        di = get_directory_from_session(processedDataDir, currSession)
        # Creates a dictionary with all the output from main_preprocess.
        data = load_grating_data(di)

        saveDir = os.path.join(
            saveDirBase, currSession["Name"], currSession["Date"])

        # Makes save dir for later.
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)

        print("getting aligned signal")
        gratingRes, ts = get_calcium_aligned(
            data["sig"],
            data["calTs"].reshape(-1, 1),
            data["gratingsSt"],
            np.array([-1, 4]).reshape(-1, 1).T,
            data["planes"].reshape(-1, 1),
            data["planeDelays"].reshape(1, -1)
        )

        print("getting trial classification")
        quietI, activeI = get_trial_classification_running(
            data["wheelVelocity"],
            data["wheelTs"],
            data["gratingsSt"],
            data["gratingsEt"],
            activeVelocity=ops["active_velocity"],
            quietVelocity=ops["quiet_velocity"],
            fractionToTest=ops["fraction_to_test"],
            criterion=ops["criterion"],
        )
    except:
        print(
            f"Could not loads basic files needed for session\n {currSession}")
        print(traceback.format_exc())
        continue
    print(f'Working on {currSession}')
    respP = np.zeros(gratingRes.shape[-1]) * np.nan
    respDirection = np.zeros(gratingRes.shape[-1]) * np.nan

    paramsOri = np.zeros((gratingRes.shape[-1], 5)) * np.nan
    # paramsOriSplit = np.zeros((7, gratingRes.shape[-1]))
    paramsOriSplit = np.zeros((gratingRes.shape[-1], 5, 2)) * np.nan
    varSpecificOri = np.zeros((gratingRes.shape[-1], 2)) * np.nan
    varOriConst = np.zeros(gratingRes.shape[-1]) * np.nan
    varOriOne = np.zeros(gratingRes.shape[-1]) * np.nan
    varOriSplit = np.zeros(gratingRes.shape[-1]) * np.nan
    pvalOri = np.zeros(gratingRes.shape[-1]) * np.nan
    TunersOri = np.empty((gratingRes.shape[-1], 2), dtype=object)
    paramsDistOri = np.zeros((gratingRes.shape[-1], 5, 2, 500)) * np.nan

    paramsTf = np.zeros((gratingRes.shape[-1], 4)) * np.nan
    # paramsTfSplit = np.zeros((8, gratingRes.shape[-1]))
    paramsTfSplit = np.zeros((gratingRes.shape[-1], 4, 2)) * np.nan
    # varsTf = np.zeros((3, gratingRes.shape[-1]))
    varTfConst = np.zeros(gratingRes.shape[-1]) * np.nan
    varTfOne = np.zeros(gratingRes.shape[-1]) * np.nan
    varTfSplit = np.zeros(gratingRes.shape[-1]) * np.nan
    varSpecificTf = np.zeros((gratingRes.shape[-1], 4)) * np.nan
    pvalTf = np.zeros(gratingRes.shape[-1]) * np.nan
    TunersTf = np.empty((gratingRes.shape[-1], 2), dtype=object)
    paramsDistTf = np.zeros((gratingRes.shape[-1], 4, 2, 500)) * np.nan

    paramsSf = np.zeros((gratingRes.shape[-1], 4)) * np.nan
    # paramsSfSplit = np.zeros((8, gratingRes.shape[-1]))
    paramsSfSplit = np.zeros((gratingRes.shape[-1], 4, 2)) * np.nan
    # varsSf = np.zeros((3, gratingRes.shape[-1]))
    varSfConst = np.zeros(gratingRes.shape[-1]) * np.nan
    varSfOne = np.zeros(gratingRes.shape[-1]) * np.nan
    varSfSplit = np.zeros(gratingRes.shape[-1]) * np.nan
    varSpecificSf = np.zeros((gratingRes.shape[-1], 4)) * np.nan
    pvalSf = np.zeros(gratingRes.shape[-1]) * np.nan
    TunersSf = np.empty((gratingRes.shape[-1], 2), dtype=object)
    paramsDistSf = np.zeros((gratingRes.shape[-1], 4, 2, 500)) * np.nan

    paramsCon = np.zeros((gratingRes.shape[-1], 4)) * np.nan
    # paramsConSplit = np.zeros((6, gratingRes.shape[-1]))
    paramsConSplit = np.zeros((gratingRes.shape[-1], 4, 2)) * np.nan
    # varsCon = np.zeros((3, gratingRes.shape[-1]))
    varConConst = np.zeros(gratingRes.shape[-1]) * np.nan
    varConOne = np.zeros(gratingRes.shape[-1]) * np.nan
    varConSplit = np.zeros(gratingRes.shape[-1]) * np.nan
    varSpecificCon = np.zeros((gratingRes.shape[-1], 4)) * np.nan
    pvalCon = np.zeros(gratingRes.shape[-1]) * np.nan
    TunersCon = np.empty((gratingRes.shape[-1], 2), dtype=object)
    paramsDistCon = np.zeros((gratingRes.shape[-1], 4, 2, 500)) * np.nan

    fittingRange = range(0, gratingRes.shape[-1])
    # check if want to run only some neurons

    if type(currSession['SpecificNeurons']) is str:
        currSession["SpecificNeurons"] = eval(currSession["SpecificNeurons"])
        fittingRange = currSession["SpecificNeurons"]
        # assume to wants to redo only those, so try reloading existing data first
        try:
            respP = np.load(os.path.join(saveDir, "resp.pval.npy"))
            respDirection = np.load(
                os.path.join(saveDir, "resp.direction.npy")
            )

            paramsOri = np.load(os.path.join(saveDir, "fit.ori.params.npy"))
            paramsOriSplit = np.load(
                os.path.join(saveDir, "fit.ori.split.params.npy")
            )
            varsOri = np.load(os.path.join(saveDir, "fit.ori.vars.npy"))
            pvalOri = np.load(os.path.join(saveDir, "fit.ori.pval.npy"))

            paramsTf = np.load(os.path.join(saveDir, "fit.tf.params.npy"))
            paramsTfSplit = np.load(
                os.path.join(saveDir, "fit.tf.split.params.npy")
            )
            varsTf = np.load(os.path.join(saveDir, "fit.tf.vars.npy"))
            pvalTf = np.load(os.path.join(saveDir, "fit.tf.pval.npy"))

            paramsSf = np.load(os.path.join(saveDir, "fit.sf.params.npy"))
            paramsSfSplit = np.load(
                os.path.join(saveDir, "fit.sf.split.params.npy")
            )
            varsSf = np.load(os.path.join(saveDir, "fit.sf.vars.npy"))
            pvalSf = np.load(os.path.join(saveDir, "fit.sf.pval.npy"))

            paramsCon = np.load(os.path.join(saveDir, "fit.con.params.npy"))
            paramsConSplit = np.load(
                os.path.join(saveDir, "fit.con.split.params.npy")
            )
            varsCon = np.load(os.path.join(saveDir, "fit.con.vars.npy"))
            pvalCon = np.load(os.path.join(saveDir, "fit.con.pval.npy"))
        except:
            pass

    blinkTrials = remove_blinking_trials(data)

    for n in fittingRange:
        print(f"Neuron {n}")
        try:
            sig, res_ori, res_freq, res_spatial, res_con = run_complete_analysis(
                gratingRes, data, ts, quietI, activeI, blinkTrials, n, ops[
                    "fitOri"], ops["fitTf"], ops["fitSf"], ops["fitContrast"]
            )

            respP[n] = sig[0]
            respDirection[n] = sig[1]

            paramsOri[n, :] = res_ori[0]

            paramsOriSplit[n, :, 0] = res_ori[1][[0, 2, 4, 5, 6]] if (
                not np.any(np.isnan(res_ori[1]))) else np.nan
            paramsOriSplit[n, :, 1] = res_ori[1][[1, 3, 4, 5, 6]] if (
                not np.any(np.isnan(res_ori[1]))) else np.nan
            # varsOri[:, n] = res_ori[2:5]
            varOriConst[n] = res_ori[2]
            varOriOne[n] = res_ori[3]
            varOriSplit[n] = res_ori[4]
            varSpecificOri[n] = res_ori[-1] if (
                not np.any(np.isnan(res_ori[-1]))) else np.ones(2) * np.nan
            pvalOri[n] = res_ori[6]
            paramsDistOri[n, :, 0, :] = res_ori[7][:, [0, 2, 4, 5, 6]].T if (
                not np.any(np.isnan(res_ori[7]))) else np.nan
            paramsDistOri[n, :, 1, :] = res_ori[7][:, [1, 3, 4, 5, 6]].T if (
                not np.any(np.isnan(res_ori[7]))) else np.nan

            paramsTf[n, :] = res_freq[0]
            paramsTfSplit[n, :, 0] = res_freq[1][::2] if (
                not np.any(np.isnan(res_freq[1]))) else np.nan
            paramsTfSplit[n, :, 1] = res_freq[1][1::2] if (
                not np.any(np.isnan(res_freq[1]))) else np.nan
            # varsTf[:, n] = res_freq[2:5]
            varTfConst[n] = res_freq[2]
            varTfOne[n] = res_freq[3]
            varTfSplit[n] = res_freq[4]
            varSpecificTf[n] = res_freq[-1] if (
                not np.any(np.isnan(res_freq[-1]))) else np.ones(4) * np.nan
            pvalTf[n] = res_freq[6]
            paramsDistTf[n, :, 0, :] = res_freq[7][:, ::2].T if (
                not np.any(np.isnan(res_freq[7]))) else np.nan
            paramsDistTf[n, :, 1, :] = res_freq[7][:, 1::2].T if (
                not np.any(np.isnan(res_freq[7]))) else np.nan

            paramsSf[n, :] = res_spatial[0]
            paramsSfSplit[n, :, 0] = res_spatial[1][::2] if (
                not np.any(np.isnan(res_spatial[1]))) else np.nan
            paramsSfSplit[n, :, 1] = res_spatial[1][1::2] if (
                not np.any(np.isnan(res_spatial[1]))) else np.nan
            # varsSf[:, n] = res_spatial[2:5]
            varSfConst[n] = res_spatial[2]
            varSfOne[n] = res_spatial[3]
            varSfSplit[n] = res_spatial[4]
            varSpecificSf[n] = res_spatial[-1] if (
                not np.all(np.isnan(res_spatial[-1]))) else np.ones(4) * np.nan
            pvalSf[n] = res_spatial[6]
            paramsDistSf[n, :, 0, :] = res_spatial[7][:, ::2].T if (
                not np.any(np.isnan(res_spatial[7]))) else np.nan
            paramsDistSf[n, :, 1, :] = res_spatial[7][:, 1::2].T if (
                not np.any(np.isnan(res_spatial[7]))) else np.nan

            paramsCon[n, :] = res_con[0]
            paramsConSplit[n, :, 0] = res_con[1][[0, 2, 4, 6]] if (
                not np.any(np.isnan(res_con[1]))) else np.nan
            paramsConSplit[n, :, 1] = res_con[1][[1, 3, 5, 7]] if (
                not np.any(np.isnan(res_con[1]))) else np.nan
            # varsCon[:, n] = res_con[2:5]
            varConConst[n] = res_con[2]
            varConOne[n] = res_con[3]
            varConSplit[n] = res_con[4]
            varSpecificCon[n] = res_con[-1] if (
                not np.any(np.isnan(res_con[-1]))) else np.ones(4) * np.nan
            pvalCon[n] = res_con[6]
            paramsDistCon[n, :, 0, :] = res_con[7][:, [0, 2, 4, 6]].T if (
                not np.any(np.isnan(res_con[7]))) else np.nan
            paramsDistCon[n, :, 1, :] = res_con[7][:, [1, 3, 5, 7]].T if (
                not np.any(np.isnan(res_con[7]))) else np.nan

        except Exception:
            print("fail " + str(n))
            print(traceback.format_exc())

    np.save(os.path.join(saveDir, "gratingResp.pVal.npy"), respP)
    np.save(os.path.join(saveDir, "gratingResp.direction.npy"), respDirection)

    if (ops["fitOri"]):
        np.save(os.path.join(
            saveDir, "gratingOriTuning.params.npy"), paramsOri)
        np.save(os.path.join(
            saveDir, "gratingOriTuning.paramsRunning.npy"), paramsOriSplit)
        np.save(os.path.join(
            saveDir, "gratingOriTuning.expVar.constant.npy"), varOriConst)
        np.save(os.path.join(
            saveDir, "gratingOriTuning.expVar.noSplit.npy"), varOriOne)
        np.save(os.path.join(
            saveDir, "gratingOriTuning.expVar.runningSplit.npy"), varOriSplit)
        np.save(os.path.join(
            saveDir, "gratingOriTuning.expVar.runningSplitSpecific.npy"), varSpecificOri)
        np.save(os.path.join(
            saveDir, "gratingOriTuning.pVal.runningSplit.npy"), pvalOri)
        np.save(os.path.join(
            saveDir, "gratingOriTuning.pVal.paramsRunningNullDist.npy"), paramsDistOri)

    if (ops["fitTf"]):
        np.save(os.path.join(saveDir, "gratingTfTuning.params.npy"), paramsTf)
        np.save(os.path.join(
            saveDir, "gratingTfTuning.paramsRunning.npy"), paramsTfSplit)
        np.save(os.path.join(
            saveDir, "gratingTfTuning.expVar.constant.npy"), varTfConst)
        np.save(os.path.join(saveDir, "gratingTfTuning.expVar.noSplit.npy"), varTfOne)
        np.save(os.path.join(
            saveDir, "gratingTfTuning.expVar.runningSplit.npy"), varTfSplit)
        np.save(os.path.join(
            saveDir, "gratingTfTuning.expVar.runningSplitSpecific.npy"), varSpecificTf)
        np.save(os.path.join(
            saveDir, "gratingTfTuning.pVal.runningSplit.npy"), pvalTf)
        np.save(os.path.join(
            saveDir, "gratingTfTuning.pVal.paramsRunningNullDist.npy"), paramsDistTf)

    if (ops["fitSf"]):
        np.save(os.path.join(saveDir, "gratingSfTuning.params.npy"), paramsSf)
        np.save(os.path.join(
            saveDir, "gratingSfTuning.paramsRunning.npy"), paramsSfSplit)
        np.save(os.path.join(
            saveDir, "gratingSfTuning.expVar.constant.npy"), varSfConst)
        np.save(os.path.join(saveDir, "gratingSfTuning.expVar.noSplit.npy"), varSfOne)
        np.save(os.path.join(
            saveDir, "gratingSfTuning.expVar.runningSplit.npy"), varSfSplit)
        np.save(os.path.join(
            saveDir, "gratingSfTuning.expVar.runningSplitSpecific.npy"), varSpecificSf)
        np.save(os.path.join(
            saveDir, "gratingSfTuning.pVal.runningSplit.npy"), pvalSf)
        np.save(os.path.join(
            saveDir, "gratingSfTuning.pVal.paramsRunningNullDist.npy"), paramsDistSf)

    if (ops["fitContrast"]):
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.params.npy"), paramsCon)
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.paramsRunning.npy"), paramsConSplit)
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.expVar.constant.npy"), varConConst)
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.expVar.noSplit.npy"), varConOne)
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.expVar.runningSplit.npy"), varConSplit)
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.expVar.runningSplitSpecific.npy"), varSpecificCon)
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.pVal.runningSplit.npy"), pvalCon)
        np.save(os.path.join(
            saveDir, "gratingContrastTuning.pVal.paramsRunningNullDist.npy"), paramsDistCon)
     # plotting
    if (ops['plot']):
        for n in fittingRange:
            try:
                print_fitting_data(gratingRes, ts, quietI, activeI, data, paramsOri,
                                   paramsOriSplit, np.vstack((varOriConst, varOriOne, varOriSplit)).T, varSpecificOri, pvalOri, paramsTf, paramsTfSplit, np.vstack(
                                       (varTfConst, varTfOne, varTfSplit)).T, varSpecificTf,
                                   pvalTf, paramsSf, paramsSfSplit, np.vstack((varSfConst, varSfOne, varSfSplit)).T, varSpecificSf, pvalSf, paramsCon, paramsConSplit, np.vstack((varConConst, varConOne, varConSplit)).T, varSpecificCon, pvalCon, n, respP, respDirection[n], None, saveDir)

            except Exception:
                print("fail " + str(n))
                print(traceback.format_exc())
        plt.close('all')

    # %%plotting
try:

    print_fitting_data(gratingRes, ts, quietI, activeI, data, paramsOri,
                       paramsOriSplit, np.vstack((varOriConst, varOriOne, varOriSplit)).T, pvalOri, paramsTf, paramsTfSplit, np.vstack(
                           (varTfConst, varTfOne, varTfSplit)).T,
                       pvalTf, paramsSf, paramsSfSplit, np.vstack((varSfConst, varSfOne, varSfSplit)).T, pvalSf, paramsCon, paramsConSplit, np.vstack((varConConst, varConOne, varConSplit)).T, pvalCon, n, respP, respDirection[n], None, saveDir)


except Exception:
    print("fail " + str(n))
    print(traceback.format_exc())
plt.close('all')

# %%
