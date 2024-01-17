# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:23:39 2023

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

from alignment_functions import get_calcium_aligned, align_stim

from fitting_classes import OriTuner, FrequencyTuner, ContrastTuner

import traceback
from abc import ABC, abstractmethod
import inspect


def get_directory_from_session(mainDir, session):
    di = os.path.join(
        mainDir, session["Name"], session["Date"], "PreprocessedFiles"
    )
    return di


# @jit(target_backend="cuda", forceobj=True)
def get_trial_classification_running(
    wheelVelocity,
    wheelTs,
    stimSt,
    stimEt,
    quietVelocity=0.5,
    activeVelocity=2,
    criterion=0.9,
):
    wheelVelocity = np.abs(wheelVelocity)
    wh, ts = align_stim(
        wheelVelocity,
        wheelTs,
        stimSt,
        np.hstack((stimSt, stimEt)) - stimSt,
    )

    whLow = wh <= quietVelocity
    whLow = np.sum(whLow, 0) / whLow.shape[0]    
    whHigh = wh >= activeVelocity
    whHigh = np.sum(whHigh[: int(whHigh.shape[0] / 2), :, 0], 0) / int(
        whHigh.shape[0] / 2
    )
    quietTrials = np.where(whLow >= criterion)[0]
    activeTrials = np.where(whHigh > 0.9)[0]
    return quietTrials, activeTrials


def make_neuron_db(
    resp,
    ts,
    quiet,
    active,
    data,
    n,
    blTime=-0.5,
):
    tf = data["gratingsTf"]
    sf = data["gratingsSf"]
    contrast = data["gratingsContrast"]
    ori = data["gratingsOri"]
    duration = data["gratingsEt"] - data["gratingsSt"]
    resp = resp[:, :, n]
    maxTime = np.min(duration)
    # trials X

    bl = np.nanmean(resp[(ts >= blTime) & (ts <= 0), :], axis=0)
    resp_corrected = resp - bl
    avg = np.zeros(resp.shape[1])
    avg_corrected = avg.copy()
    for i in range(resp.shape[1]):
        avg[i] = np.nanmean(resp[(ts > 0) & (ts <= duration[i]), i], axis=0)
        avg_corrected[i] = np.nanmean(
            resp_corrected[(ts > 0) & (ts <= duration[i]), i], axis=0
        )
    # avg = np.nanmean(resp[(ts > 0) & (ts <= maxTime)], axis=0)
    # avg_corrected = np.nanmean(
    #     resp_corrected[(ts > 0) & (ts <= maxTime)], axis=0
    # )

    movement = np.ones(avg.shape[0]) * np.nan
    movement[quiet] = 0
    movement[active] = 1

    df = pd.DataFrame(
        {
            "ori": ori[:, 0],
            "tf": tf[:, 0],
            "sf": sf[:, 0],
            "contrast": contrast[:, 0],
            "movement": movement,
            "bl": bl,
            "avg": avg,
            "avg_corrected": avg_corrected,
        }
    )

    return df


def is_responsive_direction(df, criterion=0.05):
    """
    Determines the responsiveness and direction of response by fitting a linear model
    and performs permutation testing by shuffling the target variable and fitting the 
    model to shuffled data. 
    The function calculates a p-value based on the actual R-squared score's 
    percentile among the shuffled scores.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing experimental data.
    criterion : float, optional
        The significance threshold for responsiveness. The default is 0.05.

    Returns
    -------
    p : float
        The responsiveness p-value.
    direction : int
        The direction of responsiveness (0 if not responsive, 1 for positive, -1 for negative).
    """
    direction = 0

    a = df[["ori", "tf", "sf", "contrast"]].to_numpy()
    b = df["avg_corrected"].to_numpy()

    # Identify and remove rows with non-finite (NaN or Inf) values.
    goodInds = np.where(np.isfinite(b))[0]
    a = a[goodInds, :]
    b = b[goodInds]

    # Split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.10)

    # Fit a linear model to the training data.
    res = sp.linalg.lstsq(X_train, y_train)

    # Calculate the R-squared score for the test data.
    sklearn.metrics.r2_score

    score = r2_score(y_test, X_test @ res[0])

    shuffscore = np.zeros(500)
    for s in range(500):
        # Shuffle the target variable (b) and fit the model to the shuffled data.`
        b_ = np.random.permutation(b)
        res = sp.linalg.lstsq(a, b_)
        shuffscore[s] = r2_score(b_, a @ res[0])
        
    # Calculate the percentile rank of the actual score among shuffled scores.
    p = sp.stats.percentileofscore(shuffscore, score)
    # Convert the percentile rank to a p-value.
    
    p = (100 - p) / 100
    
    # If the p-value is below the criterion, determine the responsiveness direction.
    if p < (criterion):
        direction = np.sign(np.nanmean(df["avg_corrected"]))
    return p, direction


def filter_nonsig_orientations(df, criterion=0.05):
    dfOri = df.groupby("ori")
    pVals = np.zeros(len(dfOri))
    meanOri = np.zeros(len(dfOri))
    keys = np.array(list(dfOri.groups.keys()))
    for i, dfMini in enumerate(dfOri):
        dfMini = dfMini[1]  # get the actual db
        s, p = sp.stats.ttest_rel(dfMini["bl"], dfMini["avg"])
        pVals[i] = p * len(pVals)
        meanOri[i] = dfMini["avg_corrected"].mean()
    # df = df[df["ori"].isin(keys[pVals < criterion])]
    df = df[df["ori"] == keys[np.argmax(meanOri)]]
    return df


def run_tests(
    tunerClass,
    base_name,
    split_name,
    df,
    splitter_name,
    x_name,
    y_name,
    direction=1,
):
    props_reg = np.nan
    props_split = np.nan
    score_reg = np.nan
    score_constant = np.nan
    score_split = np.nan
    dist = np.nan
    p_split = np.nan

    tunerBase = tunerClass(base_name)

    # count number of cases
    valCounts = df[x_name].value_counts().to_numpy()
    if np.any(valCounts < 3):
        return make_empty_results(x_name)

    props_reg = tunerBase.fit(
        df[x_name].to_numpy(), direction * df[y_name].to_numpy()
    )

    # fitting failed
    if np.all(np.isnan(props_reg)):
        return make_empty_results(x_name)
    score_reg = tunerBase.loo(
        df[x_name].to_numpy(), direction * df[y_name].to_numpy()
    )

    score_constant = tunerBase.loo_constant(
        df[x_name].to_numpy(), direction * df[y_name].to_numpy()
    )

    # test for split only if function predicts better
    tunerSplit = np.nan
    if score_reg > score_constant:
        tunerSplit = tunerClass(split_name, len(df[df[splitter_name] == 0]))

        dfq = df[df[splitter_name] == 0]
        dfa = df[df[splitter_name] == 1]

        # cannot run analysis if one is empty
        if (len(dfq) == 0) | (len(dfa) == 0):
            res = make_empty_results(x_name)
            res = list(res)
            res[0] = props_reg
            res[2] = score_constant
            res[3] = score_reg
            res[7] = tunerBase
            return tuple(res)

        # count values
        qCounts = dfq[x_name].value_counts().to_numpy()
        aCounts = dfa[x_name].value_counts().to_numpy()
        totCounts = np.append(qCounts, aCounts)

        # remove from fit x vals where not enough values in either dataset
        indq = np.where(qCounts < 3)[0]
        inda = np.where(aCounts < 3)[0]
        removeInds = np.union1d(indq, inda)
        removeValues = dfq[x_name].value_counts().index.to_numpy()[removeInds]
        if (len(removeInds) / len(qCounts)) > 0.33:
            res = make_empty_results(x_name)
            res = list(res)
            res[0] = props_reg
            res[2] = score_constant
            res[3] = score_reg
            res[7] = tunerBase
            return tuple(res)
        else:
            # remove only those data points that are missing
            dfq = dfq.drop(dfq[dfq[x_name].isin(removeValues)].index)
            dfa = dfa.drop(dfa[dfa[x_name].isin(removeValues)].index)

        # if (np.any(totCounts<3)):
        #     res = make_empty_results(x_name)
        #     res = list(res)
        #     res[0] = props_reg
        #     res[2] = score_constant
        #     res[3] = score_reg
        #     res[7] = tunerBase
        #     return tuple(res)

        x_sorted = np.append(dfq[x_name].to_numpy(), dfa[x_name].to_numpy())
        y_sorted = direction * np.append(
            dfq[y_name].to_numpy(), dfa[y_name].to_numpy()
        )
        props_split = tunerSplit.fit(
            x_sorted,
            y_sorted,
        )
        if np.all(np.isnan(props_split)):
            res = make_empty_results(x_name)
            res = list(res)
            res[0] = props_reg
            res[2] = score_constant
            res[3] = score_reg
            res[7] = tunerBase
            return tuple(res)
        score_split = tunerSplit.loo(
            x_sorted,
            y_sorted,
        )
        if score_split > score_reg:
            dist = tunerSplit.shuffle_split(x_sorted, y_sorted)
            p_split = sp.stats.percentileofscore(
                dist, tunerSplit.auc_diff(df[x_name].to_numpy())
            )
            # if p_split > 50:
            p_split = 100 - p_split
            p_split = p_split / 100
        else:
            p_split = np.nan
    return (
        props_reg,
        props_split,
        score_constant,
        score_reg,
        score_split,
        dist,
        p_split,
        tunerBase,
        tunerSplit,
    )


def make_empty_results(resType, *args):
    if str.lower(resType) == "ori":
        return (
            np.ones(5) * np.nan,
            np.ones(7) * np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    if (str.lower(resType) == "sf") | (str.lower(resType) == "tf"):
        return (
            np.ones(4) * np.nan,
            np.ones(8) * np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    if str.lower(resType) == "contrast":
        return (
            np.ones(4) * np.nan,
            np.ones(6) * np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    return np.nan


def run_complete_analysis(
    gratingRes,
    data,
    ts,
    quietI,
    activeI,
    n,
    runOri=True,
    runTf=True,
    runSf=True,
    runContrast=True,
):
    dfAll = make_neuron_db(
        gratingRes,
        ts,
        quietI,
        activeI,
        data,
        n,
    )
    res_ori = make_empty_results("Ori")
    res_freq = make_empty_results("Tf")
    res_spatial = make_empty_results("Sf")
    res_contrast = make_empty_results("Contrast")
    # test responsiveness
    p_resp, resp_direction = is_responsive_direction(dfAll, criterion=0.05)
    if p_resp > 0.05:
        return (
            (p_resp, resp_direction),
            res_ori,
            res_freq,
            res_spatial,
            res_contrast,
        )

    # data tests ORI
    if runOri:
        df = dfAll[
            (dfAll.sf == 0.08) & (dfAll.tf == 2) & (dfAll.contrast == 1)
        ]
        res_ori = run_tests(
            OriTuner, "gauss", "gauss_split", df, "movement", "ori", "avg"
        )

    else:
        res_ori = make_empty_results("Ori")
    #### run Tf
    if runTf:
        # Temporal Frequency tests
        df = dfAll[
            (dfAll.sf == 0.08)
            & (dfAll.tf > 0)
            & (dfAll.contrast == 1)
            & (np.isin(dfAll.ori, [0, 90, 180, 270]))
        ]

        df = filter_nonsig_orientations(df, criterion=0.05)
        res_freq = run_tests(
            FrequencyTuner, "gauss", "gauss_split", df, "movement", "tf", "avg"
        )
    else:
        res_freq = make_empty_results("Tf")
    # spatial frequency test
    if runSf:
        df = dfAll[
            (dfAll.tf == 2)
            & (dfAll.contrast == 1)
            & (np.isin(dfAll.ori, [0, 90, 180, 270]))
        ]
        df = filter_nonsig_orientations(df, criterion=0.05)
        res_spatial = run_tests(
            FrequencyTuner, "gauss", "gauss_split", df, "movement", "sf", "avg"
        )
    else:
        res_spatial = make_empty_results("Sf")
    if runContrast:
        df = dfAll[(dfAll.tf == 2) & (dfAll.sf == 0.08)]
        df = filter_nonsig_orientations(df, criterion=0.05)
        res_contrast = run_tests(
            ContrastTuner,
            "contrast",
            "contrast_split",
            df,
            "movement",
            "contrast",
            "avg",
        )

    return (
        (p_resp, resp_direction),
        res_ori,
        res_freq,
        res_spatial,
        res_contrast,
    )


def load_grating_data(directory):
    fileNameDic = {
        "sig": "calcium.dff.npy",
        "planes": "rois.planes.npy",
        "planeDelays": "planes.delay.npy",
        "calTs": "calcium.timestamps.npy",
        #"faceTs": "eye.timestamps.npy",
        "gratingsContrast": "gratings.contrast.npy",
        "gratingsOri": "gratings.direction.npy",
        "gratingsEt": "gratings.endTime.npy",
        "gratingsSt": "gratings.startTime.npy",
        # "gratingsReward": "gratings.reward.npy",
        "gratingsSf": "gratings.spatialF.npy",
        "gratingsTf": "gratings.temporalF.npy",
        "wheelTs": "wheel.timestamps.npy",
        "wheelVelocity": "wheel.velocity.npy",
    }

    # check if an update exists
    if os.path.exists(os.path.join(directory, "gratings.st.updated.npy")):
        fileNameDic["gratingsSt"] = "gratings.st.updated.npy"

    if os.path.exists(os.path.join(directory, "gratings.et.updated.npy")):
        fileNameDic["gratingsEt"] = "gratings.et.updated.npy"
    data = {}
    for key in fileNameDic.keys():
        data[key] = np.load(os.path.join(directory, fileNameDic[key]))
    return data


def load_circle_data(directory):
    fileNameDic = {
        "sig": "calcium.dff.npy",
        "planes": "calcium.planes.npy",
        "planeDelays": "planes.delay.npy",
        "calTs": "calcium.timestamps.npy",
        "faceTs": "eye.timestamps.npy",
        "wheelTs": "wheel.timestamps.npy",
        "wheelVelocity": "wheel.velocity.npy",
        "circlesSt": "circles.st.npy",
        "circlesEt": "circles.et.npy",
        "circlesX": "circles.x.npy",
        "circlesY": "circles.y.npy",
        "circlesDiameter": "circles.diameter.npy",
        "circlesIsWhite": "circles.isWhite.npy",
    }
    # check if an update exists
    if os.path.exists(os.path.join(directory, "gratings.st.updated.npy")):
        fileNameDic["gratingsSt"] = "gratings.st.updated.npy"

    if os.path.exists(os.path.join(directory, "gratings.et.updated.npy")):
        fileNameDic["gratingsEt"] = "gratings.et.updated.npy"
    data = {}
    for key in fileNameDic.keys():
        data[key] = np.load(os.path.join(directory, fileNameDic[key]))
    return data
