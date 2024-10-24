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
                   :, 0], 0) / np.squeeze(np.sum(~np.isnan(wh), 0))  # int(whLow.shape[0]/fractionToTest)

    quietTrials = np.where(whLow >= criterion)[0]

    if (activeVelocity is None):
        activeTrials = np.setdiff1d(np.arange(wh.shape[1]), quietTrials)
    else:
        whHigh = wh > activeVelocity

        whHigh = np.sum(whHigh[: int(whHigh.shape[0]/fractionToTest),
                        :, 0], 0) / np.sum(~np.isnan(wh), 0)  # int(whLow.shape[0]/fractionToTest)

        activeTrials = np.where(whHigh > criterion)[0]
    return quietTrials, activeTrials

def get_trial_average_velocity(
    wheelVelocity,
    wheelTs,
    stimSt,
    stimEt,    
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

    avgV = np.nanmean(wh,0)
    return avgV


def get_running_distribution(
    wheelVelocity,
    wheelTs,
    stimSt,
    stimEt,
    binSize=None
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
    wh = wh[~np.isnan(wh)]
    if (binSize is None):
        hist, bins = np.histogram(wh.flatten())
    else:
        dat = wh.flatten()
        bins = np.arange(np.nanmin(dat), np.nanmax(dat), binSize)
        hist, bins = np.histogram(dat, bins=bins)
    return hist, bins


def get_pupil_distribution(pupil, pupilTs, stimSt, stimEt, binSize=None):
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
    pu = pu[~np.isnan(pu)]
    if (binSize is None):
        hist, bins = np.histogram(pu.flatten())
    else:
        dat = pu.flatten()
        bins = np.arange(np.nanmin(dat), np.nanmax(dat), binSize)
        hist, bins = np.histogram(dat, bins=bins)
    return hist, bins


def get_trial_classification_pupil(
    pupil,
    pupilTs,
    stimSt,
    stimEt,
    fractionToTest=1,
    criterion=1,
    medianMask=None,
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

    if (medianMask is None):
        medianDia = np.nanmedian(pu)
    else:
        medianDia = np.nanmedian(pu[:, medianMask, :])

    puLow = pu <= medianDia
    # whLow = np.sum(whLow[: int(whLow.shape[0] / 2), :, 0], 0) / int(
    #     whLow.shape[0] / 2)
    puLow = np.sum(puLow[: int(puLow.shape[0]/fractionToTest),
                   :, 0], 0) / np.squeeze(np.sum(~np.isnan(pu), 0))/fractionToTest

    quietTrials = np.where(puLow >= criterion)[0]
    quietTrials = np.intersect1d(quietTrials, medianMask)

    puHigh = pu > medianDia

    puHigh = np.sum(puHigh[: int(puHigh.shape[0]/fractionToTest),
                    :, 0], 0) / np.squeeze(np.sum(~np.isnan(pu), 0))/fractionToTest

    activeTrials = np.where(puHigh > criterion)[0]
    activeTrials = np.intersect1d(activeTrials, medianMask)

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
    bestData = bestData.dropna()
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
        actualWindows = np.hstack((data["gratingsSt"].reshape(-1, 1), data["gratingsEt"].reshape(-1, 1))
                                  ) - data["gratingsSt"].reshape(-1, 1)
        avgWindow = np.nanmedian(actualWindows, axis=0)
        pu, pts = align_stim(
            data['pupilDiameter'],
            data['pupilTs'],
            data["gratingsSt"],
            avgWindow.reshape(1, -1),
        )

        blinksPerTrial = np.sum(np.isnan(pu), axis=0)
        blinkTrials = blinksPerTrial > 0
    return np.squeeze(blinkTrials)


def run_complete_analysis(
    gratingRes,
    data,
    ts,
    quietI,
    activeI,
    ignoreTrials,
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

    dfAll = dfAll.iloc[~ignoreTrials]

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

    if os.path.exists(os.path.join(directory, "gratings.direction.updated.npy")):
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
            data[key] = np.load(os.path.join(
                directory, fileNameDic[key]))
        else:
            Warning(
                f"The file {os.path.join(directory, fileNameDic[key])} does not exist")
            continue
    return data

def reshape_grating_data(directory):
    # Check if an update exists and load, reshape, save if necessary
    if os.path.exists(os.path.join(directory, "gratings.st.updated.npy")):
        st = np.load(os.path.join(directory, "gratings.st.updated.npy"))
        st = st.reshape(st.shape[0], 1)  
        np.save(os.path.join(directory, "gratings.st.updated.npy"), st)
        print( "st done")
    if os.path.exists(os.path.join(directory, "gratings.et.updated.npy")):
        et = np.load(os.path.join(directory, "gratings.et.updated.npy"))
        et = et.reshape(et.shape[0], 1) 
        np.save(os.path.join(directory, "gratings.et.updated.npy"), et)
        
    if os.path.exists(os.path.join(directory, "gratings.temporalF.updated.npy")):
        temporalF = np.load(os.path.join(directory, "gratings.temporalF.updated.npy"))
        temporalF = temporalF.reshape(temporalF.shape[0], 1)  
        np.save(os.path.join(directory, "gratings.temporalF.updated.npy"), temporalF)
        print ("tf done")
    if os.path.exists(os.path.join(directory, "gratings.spatialF.updated.npy")):
        spatialF = np.load(os.path.join(directory, "gratings.spatialF.updated.npy"))
        spatialF = spatialF.reshape(spatialF.shape[0], 1)  
        np.save(os.path.join(directory, "gratings.spatialF.updated.npy"), spatialF)
    
    if os.path.exists(os.path.join(directory, "gratings.ori.updated.npy")):
        ori = np.load(os.path.join(directory, "gratings.ori.updated.npy"))
        ori = ori.reshape(ori.shape[0], 1)  
        np.save(os.path.join(directory, "gratings.ori.updated.npy"), ori)
    
    if os.path.exists(os.path.join(directory, "gratings.contrast.updated.npy")):
        contrast = np.load(os.path.join(directory, "gratings.contrast.updated.npy"))
        contrast = contrast.reshape(contrast.shape[0], 1) 
        
        
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


def fit_exponential(ts, puE):
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

    puEScaled = puE[:, :].copy()
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


def get_pupil_exponential_decay(pupilTs, pupil, wheelTs, velocity, runTh=0.1, velTh=1, durTh=2, sepTh=5, decayTh=0.9, plot=False):

    pupiFreq = int(1/np.nanmedian(np.diff(pupilTs, axis=0)))
    fpupil = sp.interpolate.interp1d(
        pupilTs[:, 0], pupil[:, 0], fill_value='extrapolate')
    wheelTs = wheelTs[:len(velocity)]
    pupil = fpupil(wheelTs).reshape(-1, 1)
    pupilTs = wheelTs
    ts = wheelTs
    pupil = sp.signal.medfilt(pupil, (pupiFreq*5+1, 1))

    runningStartThreshold = (np.abs(velocity) > runTh).astype(int)
    runDiff = np.diff(runningStartThreshold, axis=0)
    runStInds = np.where(runDiff == 1)[0]
    runEtInds = np.where(runDiff == -1)[0]

    # make sure that first start is before first end
    # if there is an end before the start at the beginning ignore first end
    velocity[velocity < 0] = 0
    runningStartThreshold = (np.abs(velocity) > runTh).astype(int)
    runDiff = np.diff(runningStartThreshold, axis=0)
    runStInds = np.where(runDiff == 1)[0]
    runEtInds = np.where(runDiff == -1)[0]

    # make sure that first start is before first end
    # if there is an end before the start at the beginning ignore first end
    if (runEtInds[0] < runStInds[0]):
        runStInds = np.append(np.nan, runStInds)
        # runEtInds = runEtInds[1:]

    if (runEtInds[-1] < runStInds[-1]):
        # runStInds = runStInds[:-1]
        runEtInds = np.append(runEtInds, np.nan,)

    # make sure the stop and start times are at the non-running time
    zeroInds = np.where(np.isclose(
        velocity[:, 0], 0, rtol=10**-3, atol=10**-3))[0]

    for sti, st in enumerate(runStInds):
        # get closest zero point
        lastZero = np.where(((st-zeroInds) >= 0) & (zeroInds >= runEtInds[sti-1]) & (
            zeroInds <= runStInds[min(len(runStInds)-1, sti+1)]))[0]
        if (len(lastZero) > 0):
            lastZero = lastZero[-1]
            runStInds[sti] = zeroInds[lastZero]

    for eti, et in enumerate(runEtInds):
        # get closest zero point
        firstZero = np.where(((zeroInds-et) >= 0) & (zeroInds > runStInds[eti]) & (
            zeroInds < runStInds[min(len(runStInds)-1, eti+1)]))[0]
        if (len(firstZero) > 0):
            firstZero = firstZero[0]
            runEtInds[eti] = zeroInds[firstZero]

    f, ax = plt.subplots(1)
    ax.plot(ts, velocity, 'k')
    ax.vlines(ts[runStInds[~np.isnan(runStInds)].astype(int)], 0, 20, 'green')
    ax.vlines(ts[runEtInds[~np.isnan(runEtInds)].astype(int)], 0, 20, 'red')
    f.suptitle('first pass finding')

    # remove instances where running was not really running properly
    meanVels = np.zeros_like(runStInds)
    for i, si in enumerate(runStInds):
        if (np.isnan(si)):
            si = 0
        et = runEtInds[i]
        if (np.isnan(et)):
            et = len(velocity)-1
        et = int(et)
        si = int(si)
        meanVel = np.nanmax(velocity[si:et, 0])
        meanVels[i] = meanVel

    runStInds = runStInds[meanVels > velTh]
    runEtInds = runEtInds[meanVels > velTh]

    # concatenate running periods that are very close by
    seps = ts[runStInds[1:].astype(int)]-ts[runEtInds[:-1].astype(int)]
    etDel = []
    stDel = []
    for si, sep in enumerate(seps):
        if sep < sepTh:
            etDel.append(si)
            stDel.append(si+1)

    runStInds = np.delete(runStInds, stDel)
    runEtInds = np.delete(runEtInds, etDel)

    f, ax = plt.subplots(1)
    ax.plot(ts, velocity, 'k')
    ax.vlines(ts[runStInds[~np.isnan(runStInds)].astype(int)], 0, 20, 'green')
    ax.vlines(ts[runEtInds[~np.isnan(runEtInds)].astype(int)], 0, 20, 'red')
    f.suptitle('first pass finding')

    nanInds = ~np.isnan(runStInds)

    if np.isnan(runStInds[0]):
        runStInds[0] = 0
    if np.isnan(runEtInds[-1]):
        runEtInds[-1] = len(velocity)-1

    runSt = ts[runStInds.astype(int)]
    runEt = ts[runEtInds.astype(int)]

    # remove running periods that are too short
    durs = (runEt-runSt)[:, 0]

    runSt = runSt[durs > durTh]
    runEt = runEt[durs > durTh]

    # runStInds = runStInds[nanInds].astype(int)
    # runEtInds = runEtInds[nanInds].astype(int)

    # runSt = ts[runStInds]
    # runEt = ts[runEtInds]

    # runStInds_ = runStInds.copy()
    # runEtInds_ = runEtInds.copy()

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

    tau, scaledTrace = fit_exponential(Ets, puE)

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


def get_ignored_index(sts, specficTrials, timeWindows):
    '''


    Parameters
    ----------
    sts : TYPE
        DESCRIPTION.
    specficTrials : Trials to keep. e.g. stationary trials
    timeWindows : TYPE
        windows of times to ignore. e.g. remnant of high pupil from running.

    Returns
    -------
    ignore : TYPE
        DESCRIPTION.

    '''
    ignore = np.ones(len(sts), dtype=bool)
    ignore[specficTrials] = False
    # go over time windwos and see if indices match
    for i in range(len(sts)):
        st = sts[i, 0]
        biggerThan = st > timeWindows[:, 0]
        smallerThan = st < timeWindows[:, 1]
        inWindow = biggerThan & smallerThan
        if (np.sum(inWindow) > 0):
            ignore[i] = True
    return ignore


def take_specific_trials(data, gratingRes, gratingResOff, specficTrials, timeWindows, returnNumberRemoved=False):
    # take only stationary trials
    data['gratingsContrast'] = data['gratingsContrast'][specficTrials, :]
    data['gratingsEt'] = data['gratingsEt'][specficTrials, :]
    data['gratingsOri'] = data['gratingsOri'][specficTrials, :]
    data['gratingsReward'] = data['gratingsReward'][specficTrials, :]
    data['gratingsSf'] = data['gratingsSf'][specficTrials, :]
    data['gratingsSt'] = data['gratingsSt'][specficTrials, :]
    data['gratingsTf'] = data['gratingsTf'][specficTrials, :]
    gratingRes = gratingRes[:, specficTrials, :]
    gratingResOff = gratingResOff[:, specficTrials, :]

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
    gratingResOff = gratingResOff[:, keepInd, :]
    if (returnNumberRemoved):
        data, gratingRes, gratingResOff,

    return data, gratingRes, gratingResOff, len(specficTrials)-len(keepInd)


def make_sure_dimensionality(data):
    if 'gratingsContrast' in data.keys():
        data["gratingsContrast"] = data["gratingsContrast"].reshape(-1, 1)

    if 'gratingsEt' in data.keys():
        data["gratingsEt"] = data["gratingsEt"].reshape(-1, 1)

    if 'gratingsOri' in data.keys():
        data["gratingsOri"] = data["gratingsOri"].reshape(-1, 1)

    if 'gratingsSf' in data.keys():
        data["gratingsSf"] = data["gratingsSf"].reshape(-1, 1)

    if 'gratingsSt' in data.keys():
        data["gratingsSt"] = data["gratingsSt"].reshape(-1, 1)

    if 'gratingsTf' in data.keys():
        data["gratingsTf"] = data["gratingsTf"].reshape(-1, 1)

    return data


def find_osi_dsi(paramsOri, direction):
    tuner = OriTuner('gauss')
    rng = np.arange(0, 360, 30)
    oris = np.zeros(paramsOri.shape[0], dtype=complex)
    dris = np.zeros(paramsOri.shape[0], dtype=complex)
    for i in range(len(paramsOri)):
        prms = paramsOri[i, :]
        fnc = tuner.func(rng, *prms)
        fnc[fnc <= 0] = 0
        dri = np.sum(np.exp(np.deg2rad(rng) * 1j)*(fnc/np.sum(fnc)))
        ori = np.sum(np.exp(np.deg2rad(2*rng) * 1j)*(fnc/np.sum(fnc)))
        oris[i] = ori
        dris[i] = dri
    return oris, dris

def calculate_snr(responses):
    """
    calculated the quality index, a measure of SNR

    Parameters
    ----------
    responses : TYPEW
        DESCRIPTION.

    Returns
    -------
    None.

    """
    varTime = np.nanvar(np.nanmean(responses, 1), 0)
    varTrials = np.nanmean(np.nanvar(responses, 0), 0)
    return varTime / varTrials