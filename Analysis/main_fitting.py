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
    remove_blinking_trials,
    get_pupil_exponential_decay,
    get_trial_classification_pupil,
    take_specific_trials
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

keys = sessions.keys()

sessions = sessions.to_dict('records')

sessions = [sessions[88]]
# did the user specify only specific trials to take
isSpecificTrials = 'SpecificTrials' in keys

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

        # remove unwanted trials if required
        if (isSpecificTrials):
            if type(currSession['SpecificTrials']) is str:
                specificTrials = eval(currSession['SpecificTrials'])
                gratingsIntervals = data["gratingIntervals"]

                if gratingsIntervals.shape[1] == 1:
                    # mistake reshape properly
                    gratingsIntervals = gratingsIntervals.reshape(-1, 2)
                chosenIntervals = gratingsIntervals[specificTrials, :]

                # find indices of trials in ranges
                totalInds = []
                for interval in chosenIntervals:
                    inds = np.where((data["gratingsSt"] >= interval[0]) & (
                        data["gratingsEt"] <= interval[1]))[0]
                    totalInds.append(inds)

                totalInds = np.vstack(totalInds)

                # change all possible data
                data["gratingIntervals"] = chosenIntervals
                data["gratingsContrast"] = data["gratingsContrast"][totalInds, :]
                data["gratingsEt"] = data["gratingsEt"][totalInds, :]
                data["gratingsOri"] = data["gratingsOri"][totalInds, :]
                data["gratingsSf"] = data["gratingsSf"][totalInds, :]
                data["gratingsSt"] = data["gratingsSt"][totalInds, :]
                data["gratingsTf"] = data["gratingsTf"][totalInds, :]

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
        if (ops["classification"] == "running") | (ops["classification"] == "pupil-stationary"):
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
        if (ops["classification"] == "pupil-stationary"):
            tau, decayTime, runPupilTimes, scaledTrace = get_pupil_exponential_decay(data['pupilTs'], data['pupilDiameter'].copy(
            ), data['wheelTs'], data['wheelVelocity'], runTh=0.1, velTh=1, durTh=2, interTh=5)

            data, gratingRes = take_specific_trials(
                data, gratingRes, quietI, runPupilTimes)

            quietI, activeI = get_trial_classification_pupil(
                data["pupilDiameter"],
                data["pupilTs"],
                data["gratingsSt"],
                data["gratingsEt"],
                fractionToTest=ops["fraction_to_test"],
                criterion=ops["criterion"],
            )

        if (ops["classification"] == "pupil"):
            quietI, activeI = get_trial_classification_pupil(
                data["pupilDiameter"],
                data["pupilTs"],
                data["gratingsSt"],
                data["gratingsEt"],
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
            respP = np.load(os.path.join(saveDir, "gratingResp.pVal.npy"))
            respDirection = np.load(os.path.join(
                saveDir, "gratingResp.direction.npy"))

            if (ops["fitOri"]):
                paramsOri = np.load(os.path.join(
                    saveDir, "gratingOriTuning.params.npy"))
                paramsOriSplit = np.load(os.path.join(
                    saveDir, "gratingOriTuning.paramsRunning.npy"))
                varOriConst = np.load(os.path.join(
                    saveDir, "gratingOriTuning.expVar.constant.npy"))
                varOriOne = np.load(os.path.join(
                    saveDir, "gratingOriTuning.expVar.noSplit.npy"))
                varOriSplit = np.load(os.path.join(
                    saveDir, "gratingOriTuning.expVar.runningSplit.npy"))
                varSpecificOri = np.load(os.path.join(
                    saveDir, "gratingOriTuning.expVar.runningSplitSpecific.npy"))
                pvalOri = np.load(os.path.join(
                    saveDir, "gratingOriTuning.pVal.runningSplit.npy"))
                paramsDistOri = np.load(os.path.join(
                    saveDir, "gratingOriTuning.pVal.paramsRunningNullDist.npy"))

            if (ops["fitTf"]):
                paramsTf = np.load(os.path.join(
                    saveDir, "gratingTfTuning.params.npy"))
                paramsTfSplit = np.load(os.path.join(
                    saveDir, "gratingTfTuning.paramsRunning.npy"))
                varTfConst = np.load(os.path.join(
                    saveDir, "gratingTfTuning.expVar.constant.npy"))
                varTfOne = np.load(os.path.join(
                    saveDir, "gratingTfTuning.expVar.noSplit.npy"))
                varTfSplit = np.load(os.path.join(
                    saveDir, "gratingTfTuning.expVar.runningSplit.npy"))
                varSpecificTf = np.load(os.path.join(
                    saveDir, "gratingTfTuning.expVar.runningSplitSpecific.npy"))
                pvalTf = np.load(os.path.join(
                    saveDir, "gratingTfTuning.pVal.runningSplit.npy"))
                paramsDistTf = np.load(os.path.join(
                    saveDir, "gratingTfTuning.pVal.paramsRunningNullDist.npy"))

            if (ops["fitSf"]):
                paramsSf = np.load(os.path.join(
                    saveDir, "gratingSfTuning.params.npy"))
                paramsSfSplit = np.load(os.path.join(
                    saveDir, "gratingSfTuning.paramsRunning.npy"))
                varSfConst = np.load(os.path.join(
                    saveDir, "gratingSfTuning.expVar.constant.npy"))
                varSfOne = np.load(os.path.join(
                    saveDir, "gratingSfTuning.expVar.noSplit.npy"))
                varSfSplit = np.load(os.path.join(
                    saveDir, "gratingSfTuning.expVar.runningSplit.npy"))
                varSpecificSf = np.load(os.path.join(
                    saveDir, "gratingSfTuning.expVar.runningSplitSpecific.npy"))
                pvalSf = np.load(os.path.join(
                    saveDir, "gratingSfTuning.pVal.runningSplit.npy"))
                paramsDistSf = np.load(os.path.join(
                    saveDir, "gratingSfTuning.pVal.paramsRunningNullDist.npy"))

            if (ops["fitContrast"]):
                paramsCon = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.params.npy"))
                paramsConSplit = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.paramsRunning.npy"))
                varConConst = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.expVar.constant.npy"))
                varConOne = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.expVar.noSplit.npy"))
                varConSplit = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.expVar.runningSplit.npy"))
                varSpecificCon = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.expVar.runningSplitSpecific.npy"))
                pvalCon = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.pVal.runningSplit.npy"))
                paramsDistCon = np.load(os.path.join(
                    saveDir, "gratingContrastTuning.pVal.paramsRunningNullDist.npy"))
        except:
            print('could not load existing data')
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

    if (ops["classification"] == "running"):
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
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.noSplit.npy"), varTfOne)
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
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.noSplit.npy"), varSfOne)
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

    if (ops["classification"] == "pupil"):
        if (ops["fitOri"]):
            np.save(os.path.join(
                saveDir, "gratingOriTuning.params.npy"), paramsOri)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.paramsPupil.npy"), paramsOriSplit)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.constant.npy"), varOriConst)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.noSplit.npy"), varOriOne)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.pupilSplit.npy"), varOriSplit)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.pupilSplitSpecific.npy"), varSpecificOri)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.pVal.pupilSplit.npy"), pvalOri)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.pVal.paramsPupilNullDist.npy"), paramsDistOri)

        if (ops["fitTf"]):
            np.save(os.path.join(saveDir, "gratingTfTuning.params.npy"), paramsTf)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.paramsPupil.npy"), paramsTfSplit)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.constant.npy"), varTfConst)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.noSplit.npy"), varTfOne)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.pupilSplit.npy"), varTfSplit)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.pupilSplitSpecific.npy"), varSpecificTf)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.pVal.pupilSplit.npy"), pvalTf)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.pVal.paramsPupilNullDist.npy"), paramsDistTf)

        if (ops["fitSf"]):
            np.save(os.path.join(saveDir, "gratingSfTuning.params.npy"), paramsSf)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.paramsPupil.npy"), paramsSfSplit)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.constant.npy"), varSfConst)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.noSplit.npy"), varSfOne)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.pupilSplit.npy"), varSfSplit)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.pupilSplitSpecific.npy"), varSpecificSf)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.pVal.pupilSplit.npy"), pvalSf)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.pVal.paramsPupilNullDist.npy"), paramsDistSf)

        if (ops["fitContrast"]):
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.params.npy"), paramsCon)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.paramsPupil.npy"), paramsConSplit)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.constant.npy"), varConConst)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.noSplit.npy"), varConOne)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.pupilSplit.npy"), varConSplit)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.pupilSplitSpecific.npy"), varSpecificCon)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.pVal.pupilSplit.npy"), pvalCon)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.pVal.paramsPupilNullDist.npy"), paramsDistCon)

    if (ops["classification"] == "pupil-stationary"):
        if (ops["fitOri"]):
            np.save(os.path.join(
                saveDir, "gratingOriTuning.stationary.params.npy"), paramsOri)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.paramsPupilStationary.npy"), paramsOriSplit)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.stationay.constant.npy"), varOriConst)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.stationay.noSplit.npy"), varOriOne)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.pupilStationarySplit.npy"), varOriSplit)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.expVar.pupilStationarySplitSpecific.npy"), varSpecificOri)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.pVal.pupilStationarySplit.npy"), pvalOri)
            np.save(os.path.join(
                saveDir, "gratingOriTuning.pVal.paramsPupilStationaryNullDist.npy"), paramsDistOri)

        if (ops["fitTf"]):
            np.save(os.path.join(
                saveDir, "gratingTfTuning.stationary.params.npy"), paramsTf)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.paramsPupilStationary.npy"), paramsTfSplit)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.stationay.constant.npy"), varTfConst)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.stationay.noSplit.npy"), varTfOne)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.pupilStationarySplit.npy"), varTfSplit)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.expVar.pupilStationarySplitSpecific.npy"), varSpecificTf)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.pVal.pupilStationarySplit.npy"), pvalTf)
            np.save(os.path.join(
                saveDir, "gratingTfTuning.pVal.paramsPupilStationaryNullDist.npy"), paramsDistTf)

        if (ops["fitSf"]):
            np.save(os.path.join(
                saveDir, "gratingSfTuning.stationary.params.npy"), paramsSf)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.paramsPupilStationary.npy"), paramsSfSplit)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.stationay.constant.npy"), varSfConst)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.stationay.noSplit.npy"), varSfOne)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.pupilStationarySplit.npy"), varSfSplit)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.expVar.pupilStationarySplitSpecific.npy"), varSpecificSf)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.pVal.pupilStationarySplit.npy"), pvalSf)
            np.save(os.path.join(
                saveDir, "gratingSfTuning.pVal.paramsPupilStationaryNullDist.npy"), paramsDistSf)

        if (ops["fitContrast"]):
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.stationary.params.npy"), paramsCon)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.paramsPupilStationary.npy"), paramsConSplit)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.stationay.constant.npy"), varConConst)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.stationay.noSplit.npy"), varConOne)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.pupilStationarySplit.npy"), varConSplit)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.expVar.pupilStationarySplitSpecific.npy"), varSpecificCon)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.pVal.pupilStationarySplit.npy"), pvalCon)
            np.save(os.path.join(
                saveDir, "gratingContrastTuning.pVal.paramsPupilStationaryNullDist.npy"), paramsDistCon)

     # plotting
    if (ops['plot']):
        for n in fittingRange:
            try:
                print_fitting_data(gratingRes, ts, quietI, activeI, data, paramsOri,
                                   paramsOriSplit, np.vstack((varOriConst, varOriOne, varOriSplit)).T, varSpecificOri, pvalOri, paramsTf, paramsTfSplit, np.vstack(
                                       (varTfConst, varTfOne, varTfSplit)).T, varSpecificTf,
                                   pvalTf, paramsSf, paramsSfSplit, np.vstack((varSfConst, varSfOne, varSfSplit)).T, varSpecificSf, pvalSf, paramsCon, paramsConSplit, np.vstack((varConConst, varConOne, varConSplit)).T, varSpecificCon, pvalCon, n, respP, respDirection[n], None, saveDir, subDir=ops['classification'])

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
