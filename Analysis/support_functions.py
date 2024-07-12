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
        mainDir, session["Name"], session["Date"],
    )
    return di


# @jit(target_backend="cuda", forceobj=True)
def get_trial_classification_running(
    wheelVelocity,
    wheelTs,
    stimSt,
    stimEt,
    quietVelocity=0.5,
    activeVelocity=None,
    fractionToTest=1,
    criterion=1,
):
    wheelVelocity = np.abs(wheelVelocity)

    stimSt = stimSt.reshape(-1, 1)
    stimEt = stimEt.reshape(-1, 1)

    wh, ts = align_stim(
        wheelVelocity,
        wheelTs,
        stimSt,
        np.hstack((stimSt, stimEt)) - stimSt,
    )

    whLow = wh <= quietVelocity
    # whLow = np.sum(whLow[: int(whLow.shape[0] / 2), :, 0], 0) / int(
    #     whLow.shape[0] / 2)
    whLow = np.sum(whLow[: int(whLow.shape[0]/fractionToTest),
                   :, 0], 0) / int(whLow.shape[0]/fractionToTest)

    quietTrials = np.where(whLow >= criterion)[0]

    if (activeVelocity is None):
        activeTrials = np.setdiff1d(np.arange(wh.shape[1]), quietTrials)
    else:
        whHigh = wh > activeVelocity

        whHigh = np.sum(whHigh[: int(whHigh.shape[0]/fractionToTest),
                        :, 0], 0) / int(whLow.shape[0]/fractionToTest)

        activeTrials = np.where(whHigh > criterion)[0]
    return quietTrials, activeTrials


def get_trial_classification_pupil(
    pupil,
    pupilTs,
    stimSt,
    stimEt,
    fractionToTest=1,
    criterion=1,
):

    pupiFreq = int(1/np.nanmedian(np.diff(pupilTs, axis=0)))
    pupil = sp.signal.medfilt(pupil, (pupiFreq*5+1, 1))

    stimSt = stimSt.reshape(-1, 1)
    stimEt = stimEt.reshape(-1, 1)

    pu, ts = align_stim(
        pupil,
        pupilTs,
        stimSt,
        np.hstack((stimSt, stimEt)) - stimSt,
    )

    medianDia = np.nanmedian(pu)

    puLow = pu <= medianDia
    # whLow = np.sum(whLow[: int(whLow.shape[0] / 2), :, 0], 0) / int(
    #     whLow.shape[0] / 2)
    puLow = np.sum(puLow[: int(puLow.shape[0]/fractionToTest),
                   :, 0], 0) / int(puLow.shape[0]/fractionToTest)

    quietTrials = np.where(puLow >= criterion)[0]

    puHigh = pu > medianDia

    puHigh = np.sum(puHigh[: int(puHigh.shape[0]/fractionToTest),
                    :, 0], 0) / int(puHigh.shape[0]/fractionToTest)

    activeTrials = np.where(puHigh > criterion)[0]
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

    # group to find best values
    dGroups = df.groupby(['ori', 'tf', 'sf', 'contrast'])
    groupCounts = dGroups.count()
    groupMeans = dGroups.mean()
    enoughTrials = np.where(groupCounts['avg_corrected'] >= 10)[0]
    groupMeans_enough = groupMeans.iloc[enoughTrials]
    groupMeans_enough['avg_corrected'] = np.abs(
        groupMeans_enough['avg_corrected'])
    maxId = groupMeans_enough.idxmax()['avg_corrected']

    bestData = dGroups.get_group(maxId)

    _, p = sp.stats.ttest_rel(bestData['bl'], bestData['avg'])

    if p < (criterion):
        direction = np.sign(np.nanmean(bestData["avg_corrected"]))
    return p, direction


def filter_nonsig_orientations(df, direction=1, criterion=0.05):
    dfOri = df.groupby("ori")
    pVals = np.zeros(len(dfOri))
    meanOri = np.zeros(len(dfOri))
    keys = np.array(list(dfOri.groups.keys()))
    for i, dfMini in enumerate(dfOri):
        dfMini = dfMini[1]  # get the actual db
        s, p = sp.stats.ttest_rel(dfMini["bl"], dfMini["avg"])
        pVals[i] = p * len(pVals)
        meanOri[i] = direction*dfMini["avg_corrected"].mean()
    # df = df[df["ori"].isin(keys[pVals < criterion])]
    # df = df[df["ori"] == keys[np.argmax(meanOri)]]
    pks = sp.signal.find_peaks(meanOri)[0]

    if len(pks) == 0:
        pks = []

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
    split_test_inds,
    direction=1,

):
    props_reg = np.nan
    props_split = np.nan
    score_reg = np.nan
    score_constant = np.nan
    score_split = np.nan
    dist = np.nan
    p_split = np.nan
    score_split_specific = np.nan
    propsDist = np.nan

    tunerBase = tunerClass(base_name)

    # Remove all Nans and Inf
    goodInds = np.where(np.isfinite(df[y_name]))[0]
    df = df.iloc[goodInds]

    # enough test cases
    if (len(np.unique(df[x_name]))) <= 2:
        return make_empty_results(x_name)
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
            return tuple(res)

        # count values
        qCounts = dfq[x_name].value_counts().to_numpy()
        aCounts = dfa[x_name].value_counts().to_numpy()
        totCounts = np.append(qCounts, aCounts)

        # remove from fit x vals where not enough values in either dataset
        indq = np.where(qCounts < 3)[0]
        inda = np.where(aCounts < 3)[0]
        # removeInds = np.union1d(indq, inda)
        valsQ = dfq[x_name].value_counts().index.to_numpy()
        valsA = dfa[x_name].value_counts().index.to_numpy()
        # removeValues = np.union1d(removeValuesQ, removeValuesA)

        if (len(indq) > 0) | (len(inda) > 0) | (len(valsQ) < len(np.unique(df[x_name]))) | (len(valsA) < len(np.unique(df[x_name]))):
            res = make_empty_results(x_name)
            res = list(res)
            res[0] = props_reg
            res[2] = score_constant
            res[3] = score_reg
            return tuple(res)

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
            return tuple(res)
        score_split = tunerSplit.loo(
            x_sorted,
            y_sorted,
        )
        if score_split > score_reg:
            dist, propsDist = tunerSplit.shuffle_split(
                x_sorted, y_sorted, returnNull=True)
            p_split = sp.stats.percentileofscore(
                dist, tunerSplit.auc_diff(df[x_name].to_numpy())
            )
            # if p_split > 50:
            p_split = 100 - p_split
            p_split = p_split / 100

            if (p_split <= 0.05):
                fixedProps = props_reg[split_test_inds.astype(int)]
                score_split_specific, propsList = tunerSplit.loo_fix_variables(
                    x_sorted, y_sorted, fixedProps)

                # ignore cases were the fixed case could not actually fit well
                score_split_specific_ = score_split_specific[~np.any(
                    np.isnan(propsList), axis=1)]
                propsList_ = propsList[~np.any(np.isnan(propsList), axis=1)]

                score_split_specific[np.any(
                    np.isnan(propsList), axis=1)] = np.nan
                propsList[np.any(np.isnan(propsList), axis=1)] = np.nan

                if (len(propsList_) > 0):
                    maxFixedScore = np.max(score_split_specific_)
                    if (maxFixedScore > score_split):
                        props_split = propsList_[
                            np.argmax(score_split_specific_), :]
            else:
                score_split_specific = np.ones(len(split_test_inds))*np.nan

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
        propsDist,
        score_split_specific,

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
            np.nan

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
            np.ones(8) * np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    return np.nan


def remove_blinking_trials(data):
    blinkTrials = np.zeros(len(data["gratingsSt"]), dtype=bool)
    if ('pupilDiameter' in data.keys()):
        pu, pts = align_stim(
            data['pupilDiameter'],
            data['pupilTs'],
            data["gratingsSt"],
            np.hstack((data["gratingsSt"].reshape(-1, 1), data["gratingsEt"].reshape(-1, 1))
                      ) - data["gratingsEt"].reshape(-1, 1),
        )
        blinksPerTrial = np.sum(np.isnan(pu), axis=0)
        blinkTrials = blinksPerTrial > 0
    return blinkTrials


def run_complete_analysis(
    gratingRes,
    data,
    ts,
    quietI,
    activeI,
    blinkTrials,
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

    dfAll = dfAll.iloc[~blinkTrials]

    res_ori = make_empty_results("Ori")
    res_freq = make_empty_results("Tf")
    res_spatial = make_empty_results("Sf")
    res_contrast = make_empty_results("contrast")
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
            OriTuner, "gauss", "gauss_split", df, "movement", "ori", "avg", np.array([
                0, 1]), np.sign(df['avg'].mean())  # resp_direction
        )

    else:
        res_ori = make_empty_results("Ori")
    # run Tf
    if runTf:
        # Temporal Frequency tests
        df = dfAll[
            (dfAll.sf == 0.08)
            & (dfAll.tf > 0)
            & (dfAll.contrast == 1)
            & (np.isin(dfAll.ori, [0, 90, 180, 270]))
        ]

        df = filter_nonsig_orientations(df, resp_direction, criterion=0.05)
        res_freq = run_tests(
            FrequencyTuner, "gauss", "gauss_split", df, "movement", "tf", "avg", np.array([
                0, 1, 2, 3]), resp_direction
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
        df = filter_nonsig_orientations(df, resp_direction, criterion=0.05)
        res_spatial = run_tests(
            FrequencyTuner, "gauss", "gauss_split", df, "movement", "sf", "avg",  np.array([
                0, 1, 2, 3]), resp_direction
        )
    else:
        res_spatial = make_empty_results("Sf")
    if runContrast:
        df = dfAll[(dfAll.tf == 2) & (dfAll.sf == 0.08)]
        df = filter_nonsig_orientations(df, resp_direction, criterion=0.05)
        res_contrast = run_tests(
            ContrastTuner,
            "contrast",
            "contrast_split_full",
            df,
            "movement",
            "contrast",
            "avg",
            np.array([
                0, 1, 2, 3]),
            resp_direction
        )
    else:
        res_contrast = make_empty_results("contrast")

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
        "faceTs": "eye.timestamps.npy",
        "gratingsContrast": "gratings.contrast.npy",
        "gratingsOri": "gratings.direction.npy",
        "gratingsEt": "gratings.endTime.npy",
        "gratingsSt": "gratings.startTime.npy",
        "gratingsReward": "gratings.reward.npy",
        "gratingsSf": "gratings.spatialF.npy",
        "gratingsTf": "gratings.temporalF.npy",
        "wheelTs": "wheel.timestamps.npy",
        "wheelVelocity": "wheel.velocity.npy",
        "pupilDiameter": "eye.diameter.npy",
        "pupilTs": "eye.timestamps.npy",
        "gratingIntervals": "gratingsExp.intervals.npy",
        "RoiId": "rois.id.npy",
    }

    # check if an update exists
    if os.path.exists(os.path.join(directory, "gratings.st.updated.npy")):
        fileNameDic["gratingsSt"] = "gratings.st.updated.npy"

    if os.path.exists(os.path.join(directory, "gratings.et.updated.npy")):
        fileNameDic["gratingsEt"] = "gratings.et.updated.npy"

    if os.path.exists(os.path.join(directory, "gratings.temporalF.updated.npy")):
        fileNameDic["gratingsTf"] = "gratings.temporalF.updated.npy"

    if os.path.exists(os.path.join(directory, "gratings.spatialF.updated.npy")):
        fileNameDic["gratingsSf"] = "gratings.spatialF.updated.npy"

    if os.path.exists(os.path.join(directory, "gratings.spatialF.updated.npy")):
        fileNameDic["gratingsOri"] = "gratings.direction.updated.npy"
    if os.path.exists(os.path.join(directory, "gratings.contrast.updated.npy")):
        fileNameDic["gratingsContrast"] = "gratings.contrast.updated.npy"
    data = {}
    for key in fileNameDic.keys():
        if (key == "planes"):
            if not (os.path.exists(os.path.join(directory, fileNameDic[key]))):
                if(os.path.exists(os.path.join(directory, "calcium.planes.npy"))):
                    os.rename(os.path.join(directory, "calcium.planes.npy"), os.path.exists(
                        os.path.join(directory, fileNameDic[key])))
        if (os.path.exists(os.path.join(directory, fileNameDic[key]))):
            data[key] = np.load(os.path.join(directory, fileNameDic[key]))
        else:
            Warning(
                f"The file {os.path.join(directory, fileNameDic[key])} does not exist")
            continue
    return data


def load_circle_data(directory):
    fileNameDic = {
        "sig": "calcium.dff.npy",
        "planes": "rois.planes.npy",
        "planeDelays": "planes.delay.npy",
        "calTs": "calcium.timestamps.npy",
        "faceTs": "eye.timestamps.npy",
        "wheelTs": "wheel.timestamps.npy",
        "wheelVelocity": "wheel.velocity.npy",
        "circlesSt": "circles.startTime.npy",
        "circlesEt": "circles.endTime.npy",
        "circlesY": "circles.y.npy",
        "circlesX": "circles.x.npy",
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

        if (key == "planes"):
            if not (os.path.exists(os.path.join(directory, fileNameDic[key]))):
                if(os.path.exists(os.path.join(directory, "calcium.planes.npy"))):
                    os.rename(os.path.join(directory, "calcium.planes.npy"),
                              os.path.join(directory, fileNameDic[key]))

        data[key] = np.load(os.path.join(directory, fileNameDic[key]))
    return data


def fit_exponential(ts, puE, goodInds):
    '''
    get the time points from end of running sup to the future.
    fit a general decay and then the offset for each case

plt.plot()
    Parameters
    ----------
    ts : TYPE
        DESCRIPTION.
    puS : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    puEScaled = puE[:, goodInds].copy()
    puEScaled_m = np.nanmean(puEScaled, 1)

    puEScaled_m -= np.nanmin(puEScaled_m)
    puEScaled_m /= puEScaled_m[ts == 0]

    ydata = puEScaled_m[ts >= 0]
    xdata = ts[ts >= 0]
    A = ydata[0]
    propsDecay, _ = sp.optimize.curve_fit(
        lambda t, tau: A*np.exp(-t/tau), xdata, ydata)
    tau = propsDecay[0]

    return tau, puEScaled_m


def get_pupil_exponential_decay(pupilTs, pupil, wheelTs, velocity, runTh=0.1, velTh=1, durTh=2, interTh=5, decayTh=0.9, plot=False):
    pupiFreq = int(1/np.nanmedian(np.diff(pupilTs, axis=0)))
    fpupil = sp.interpolate.interp1d(
        pupilTs[:, 0], pupil[:, 0], fill_value='extrapolate')
    pupil = fpupil(wheelTs).reshape(-1, 1)
    pupilTs = wheelTs
    pupil = sp.signal.medfilt(pupil, (pupiFreq*5+1, 1))

    runningStartThreshold = (np.abs(velocity) > runTh).astype(int)
    runDiff = np.diff(runningStartThreshold, axis=0)
    runStInds = np.where(runDiff == 1)[0]
    runEtInds = np.where(runDiff == -1)[0]

    # make sure that first start is before first end
    # if there is an end before the start at the beginning ignore first end
    if (runEtInds[0] < runStInds[0]):
        runEtInds = runEtInds[1:]

    if (runEtInds[-1] < runStInds[-1]):
        runStInds = runStInds[:-1]

    # remove instances where running was not really fast
    meanVels = np.zeros_like(runStInds)
    for i, si in enumerate(runStInds):
        meanVel = np.nanmean(velocity[si:runEtInds[i]])
        meanVels[i] = meanVel

    # make sure running bouts are long enough and there is enough time between them

    runStInds_ = runStInds.copy()
    runEtInds_ = runEtInds.copy()
    runStInds = runStInds[meanVels > velTh]
    runEtInds = runEtInds[meanVels > velTh]

    runSt = wheelTs[runStInds]
    runEt = wheelTs[runEtInds]
    runSt_ = wheelTs[runStInds_]
    runEt_ = wheelTs[runEtInds_]

    runDurs = runEt-runSt

    runDurThresholdInd = np.where(np.floor(runDurs) > durTh)[0]
    interboutTimesSt = runSt[1:]-runEt[:-1]
    interThInd = np.where(np.floor(interboutTimesSt) > interTh)[0]
    goodIndsS = np.intersect1d(runDurThresholdInd, interThInd+1)
    goodIndsE = np.intersect1d(runDurThresholdInd, interThInd)

    whE, wts = align_stim(
        velocity,
        wheelTs,
        runEt,  # [goodIndsE],
        np.array([-0.5, 5]).reshape(1, -1),
    )

    puE, Ets = align_stim(
        pupil,
        pupilTs,
        runEt,  # [goodIndsE],
        np.array([-0.5, 10]).reshape(1, -1),
    )

    puE = np.squeeze(puE)
    puEScaled = puE.copy()
    puEScaled /= puEScaled[Ets == 0]

    tau, scaledTrace = fit_exponential(Ets, puE, goodIndsE)

    decayTime = -tau*np.log(0.1)

    expFunc = 1*np.exp(-Ets/tau)

    # make the timespans to exclude
    runPupilTimes = np.hstack((runEt, runEt+decayTime))

    if (plot):
        f, ax = plt.subplots(1)
        f.suptitle('running end')
        ax.plot(Ets, scaledTrace, 'k')
        ax.plot(Ets, expFunc, 'orange')
        ax.set_xlabel('Time from running bout end (s)')
        ax.set_ylabel('pupil diamater')
        ax.hlines(1-decayTh, Ets[0], Ets[-1], 'r', ls='dashed')

    return tau, decayTime, runPupilTimes, scaledTrace


def take_specific_trials(data, gratingRes, specficTrials, timeWindows):
    # take only stationary trials
    data['gratingsContrast'] = data['gratingsContrast'][specficTrials, :]
    data['gratingsEt'] = data['gratingsEt'][specficTrials, :]
    data['gratingsOri'] = data['gratingsOri'][specficTrials, :]
    data['gratingsReward'] = data['gratingsReward'][specficTrials, :]
    data['gratingsSf'] = data['gratingsSf'][specficTrials, :]
    data['gratingsSt'] = data['gratingsSt'][specficTrials, :]
    data['gratingsTf'] = data['gratingsTf'][specficTrials, :]
    gratingRes = gratingRes[:, specficTrials, :]

    # go over time windwos and see if indices match
    keepInd = np.zeros_like(data['gratingsSt'], dtype=bool)
    for i in range(data['gratingsSt'].shape[0]):
        st = data['gratingsSt'][i, 0]
        biggerThan = st > timeWindows[:, 0]
        smallerThan = st < timeWindows[:, 1]
        inWindow = biggerThan & smallerThan
        if (np.sum(inWindow) == 0):
            keepInd[i] = True

    keepInd = np.squeeze(keepInd)
    data['gratingsContrast'] = data['gratingsContrast'][keepInd, :]
    data['gratingsEt'] = data['gratingsEt'][keepInd, :]
    data['gratingsOri'] = data['gratingsOri'][keepInd, :]
    data['gratingsReward'] = data['gratingsReward'][keepInd, :]
    data['gratingsSf'] = data['gratingsSf'][keepInd, :]
    data['gratingsSt'] = data['gratingsSt'][keepInd, :]
    data['gratingsTf'] = data['gratingsTf'][keepInd, :]
    gratingRes = gratingRes[:, keepInd, :]

    return data, gratingRes
