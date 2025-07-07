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
    shuffleBasedFunc = None,
    xtol = None,
    ftol = None,
    Nshuffle = 500,
    

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
    
    if (xtol is None) and (ftol is None):
        tunerBase = tunerClass(base_name)
    elif (xtol is None):
        tunerBase = tunerClass(base_name,ftol=ftol)
    else:
        tunerBase = tunerClass(base_name,ftol=ftol,xtol=xtol)

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
        
        if (xtol is None) and (ftol is None):
            tunerSplit = tunerClass(split_name, len(df[df[splitter_name] == 0]))
        elif (xtol is None):
            tunerSplit = tunerClass(split_name, len(df[df[splitter_name] == 0]),ftol=ftol)
        else:
            tunerSplit = tunerClass(split_name, len(df[df[splitter_name] == 0]),ftol=ftol,xtol=xtol)

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
                x_sorted, y_sorted, returnNull=True,nshuff=Nshuffle)
            p_split = sp.stats.percentileofscore(
                dist, tunerSplit.auc_diff(df[x_name].to_numpy())
            )
            # if p_split > 50:
            p_split = 100 - p_split
            p_split = p_split / 100


            if (p_split <= 0.05):
                fixedProps = props_reg[split_test_inds.astype(int)]
                if ( shuffleBasedFunc is None):
                    score_split_specific, propsList = tunerSplit.loo_fix_variables(
                        x_sorted, y_sorted, fixedProps,startValues=props_reg)
    
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
                    score_split_specific = shuffleBasedFunc(propsDist,props_split)
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

def get_contrast_params(p):
    tuner = ContrastTuner('contrast_modified')
    crange = np.arange(0,1.01,0.01)
    curve = tuner.func(crange,*p)
    Rc = np.nanmax(curve)
    cMax = crange[np.argmax(curve)]
    cMin = crange[np.argmin(curve)]
    c50c = crange[crange<cMax][np.argmin(np.abs(curve[crange<cMax]-Rc/2))]
    
    curveDiff = np.diff(curve,prepend=True,axis=0)/0.1
    c50s = curveDiff[crange==c50c][0]
    
    return Rc,cMax,cMin,c50c,c50s


def contrast_shuffle_test(propsDist,propsReal):
    Rc1,cMax1,cMin1,c50c1,c50s1 = get_contrast_params(propsReal[::2])
    Rc2,cMax2,cMin2,c50c2,c50s2 = get_contrast_params(propsReal[1::2])
    
    Rcd = np.zeros(len(propsDist))
    cMaxd = np.zeros(len(propsDist))
    c50cd = np.zeros(len(propsDist))
    c50sd = np.zeros(len(propsDist))
    for i in range(len(propsDist)):
        Rct1,cMaxt1,cMint1,c50ct1,c50st1 = get_contrast_params(propsDist[i,::2])
        Rct2,cMaxt2,cMint2,c50ct2,c50st2 = get_contrast_params(propsDist[i,1::2])
        Rcd[i] = Rct1-Rct2
        cMaxd[i] = cMaxt1-cMaxt2
        c50cd[i] = c50ct1-c50ct2
        c50sd[i] = c50st1-c50st2
        
    Rscore = sp.stats.percentileofscore(Rcd,Rc1-Rc2)/100
    cmaxScore = sp.stats.percentileofscore(cMaxd,cMax1-cMax2)/100
    c50Score = sp.stats.percentileofscore(c50cd,c50c1-c50c2)/100
    c50sScore = sp.stats.percentileofscore(c50sd,c50s1-c50s2)/100
    
    # create two sided score
    Rscore = 1-Rscore if Rscore>0.5 else Rscore
    cmaxScore = 1-cmaxScore if cmaxScore>0.5 else cmaxScore
    c50Score = 1-c50Score if c50Score>0.5 else c50Score
    c50sScore = 1-c50sScore if c50sScore>0.5 else c50sScore
    
     # two sided score
    Rscore *=2
    cmaxScore *=2
    c50Score *=2
    c50sScore*=2
    
    return Rscore,cmaxScore,c50Score,c50sScore
        
        
        
        
    

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
    contrastType= 'regular'
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
        print ('direction analysis')
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
        print ('temporal freq analysis')
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
        print ('spatial freq analysis')
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
        if contrastType =='regular':
            conFun = 'contrast'
            conSplitFun = 'contrast_split_full'
        else:
            conFun = 'contrast_modified'
            conSplitFun = 'contrast_modified_split'
            
        print ('contrast analysis')
        df = dfAll[(dfAll.tf == 2) & (dfAll.sf == 0.08)]
        df = filter_nonsig_orientations(df, resp_direction, criterion=0.05)
        res_contrast = run_tests(
            ContrastTuner,
            conFun,#"contrast_modified",
            conSplitFun,#"contrast_modified_split",
            df,
            "movement",
            "contrast",
            "avg",
            np.array([
                0, 1, 2, 3]),
            resp_direction,
            #xtol = 10**-10,
            ftol = 10**-7, 
            shuffleBasedFunc = contrast_shuffle_test,Nshuffle=1000
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

def load_stimulus_data(directory,dataDict):
    data = {}
    for key in dataDict.keys():
        if (key == "planes"):
            if not (os.path.exists(os.path.join(directory, dataDict[key]))):
                if(os.path.exists(os.path.join(directory, "calcium.planes.npy"))):
                    os.rename(os.path.join(directory, "calcium.planes.npy"), os.path.exists(
                        os.path.join(directory, dataDict[key])))
        if (os.path.exists(os.path.join(directory, dataDict[key]))):
            data[key] = np.load(os.path.join(
                directory, dataDict[key]))
        else:
            Warning(
                f"The file {os.path.join(directory, dataDict[key])} does not exist")
            continue
    return data
    

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
    data = load_stimulus_data(directory,fileNameDic)
    return data

def load_luminance_data(directory):
    fileNameDic = {
        "sig": "calcium.dff.npy",
        "planes": "rois.planes.npy",
        "planeDelays": "planes.delay.npy",
        "calTs": "calcium.timestamps.npy",
        "faceTs": "eye.timestamps.npy",
        "Luminance": "Luminance.luminance.npy",        
        "LuminanceEt": "Luminance.endTime.npy",
        "LuminanceSt": "Luminance.startTime.npy",
        "wheelTs": "wheel.timestamps.npy",
        "wheelVelocity": "wheel.velocity.npy",
        "pupilDiameter": "eye.diameter.npy",
        "pupilTs": "eye.timestamps.npy",
        "gratingIntervals": "luminanceExp.intervals.npy",
        "RoiId": "rois.id.npy",
    }

    data = load_stimulus_data(directory,fileNameDic)
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

    puEScaled = puE.copy()
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
    rng = np.arange(0, 360, 15)
    oris = np.zeros(paramsOri.shape[0], dtype=complex)
    dris = np.zeros(paramsOri.shape[0], dtype=complex)
    for i in range(len(paramsOri)):
        prms = paramsOri[i, :]
        fnc = tuner.func(rng, *prms).astype(np.float64)
        fnc[fnc <= 0] = 0
        fnc/=np.nanmax(fnc)
        dri = np.sum(np.exp(np.deg2rad(rng) * 1j)*(fnc/np.sum(fnc)))
        ori = np.sum(np.exp(np.deg2rad(2*rng) * 1j)*(fnc/np.sum(fnc)))
        oris[i] = ori
        dris[i] = dri
    return oris, dris



def load_grating_fitting_data(saveDir,sessionCsv):
    sessions = pd.read_csv(sessionCsv)

    sessions = sessions.to_dict('records')
    
    analysisList = {"respP": "gratingResp.pVal.npy", "respDirection": "gratingResp.direction.npy",'paramsOri': 'gratingOriTuning.params.npy', 'paramsOriSplit': "gratingOriTuning.paramsRunning.npy", "varOriC": "gratingOriTuning.expVar.constant.npy", "varOriS": "gratingOriTuning.expVar.runningSplit.npy", "varOriN": "gratingOriTuning.expVar.noSplit.npy", "pvalOri": "gratingOriTuning.pVal.runningSplit.npy", "varOriSpecific": "gratingOriTuning.expVar.runningSplitSpecific.npy", "nullOri": "gratingOriTuning.pVal.paramsRunningNullDist.npy","permOri": "gratingOriTuning.pVal.permutationTest.npy",
                    'paramsTf': 'gratingTfTuning.params.npy', 'paramsTfSplit': "gratingTfTuning.paramsRunning.npy", "varTfC": "gratingTfTuning.expVar.constant.npy", "varTfS": "gratingTfTuning.expVar.runningSplit.npy", "varTfN": "gratingTfTuning.expVar.noSplit.npy", "pvalTf": "gratingTfTuning.pVal.runningSplit.npy", "varTfSpecific": "gratingTfTuning.expVar.runningSplitSpecific.npy","permTf": "gratingTfTuning.pVal.permutationTest.npy",  "nullTf": "gratingTfTuning.pVal.paramsRunningNullDist.npy",
                    'paramsSf': 'gratingSfTuning.params.npy', 'paramsSfSplit': "gratingSfTuning.paramsRunning.npy", "varSfC": "gratingSfTuning.expVar.constant.npy", "varSfS": "gratingSfTuning.expVar.runningSplit.npy", "varSfN": "gratingSfTuning.expVar.noSplit.npy", "pvalSf": "gratingSfTuning.pVal.runningSplit.npy", "varSfSpecific": "gratingSfTuning.expVar.runningSplitSpecific.npy","permSf": "gratingSfTuning.pVal.permutationTest.npy","nullSf": "gratingSfTuning.pVal.paramsRunningNullDist.npy",
                    'paramsContrast': 'gratingContrastTuning.params.npy', 'paramsContrastSplit': "gratingContrastTuning.paramsRunning.npy", "varContrastC": "gratingContrastTuning.expVar.constant.npy", "varContrastS": "gratingContrastTuning.expVar.runningSplit.npy", "varContrastN": "gratingContrastTuning.expVar.noSplit.npy", "pvalContrast": "gratingContrastTuning.pVal.runningSplit.npy", "varContrastSpecific": "gratingContrastTuning.expVar.runningSplitSpecific.npy","permContrast": "gratingContrastTuning.pVal.permutationTest.npy","nullContrast": "gratingContrastTuning.pVal.paramsRunningNullDist.npy",
                    }

    defaultShapes = {"respP": (1,), "respDirection": (1,),
                     'paramsOri': (5,), 'paramsOriSplit': (5,2,), "varOriC": (1,), "varOriS": (1,), "varOriN": (1,), "pvalOri": (1,), "varOriSpecific": (2,), "nullOri": (5,2,500,),"permOri": (5,),
                    'paramsTf': (4,), 'paramsTfSplit': (4,2,), "varTfC":(1,), "varTfS": (1,), "varTfN": (1,), "pvalTf": (1,), "varTfSpecific": (4,),"permTf": (4,),  "nullTf": (4,2,500,),
                    'paramsSf': (4,), 'paramsSfSplit': (4,2,), "varSfC": (1,), "varSfS": (1,), "varSfN": (1,), "pvalSf": (1,), "varSfSpecific": (4,),"permSf": (4,),"nullSf": (4,2,500,),
                    'paramsContrast': (4,), 'paramsContrastSplit': (4,2,), "varContrastC": (1,), "varContrastS": (1,), "varContrastN": (1,), "pvalContrast": (1,), "varContrastSpecific": (4,),"permContrast": (4,),"nullContrast": (4,2,1000,),
                        }
    analysisData = []
    for si, s in enumerate(sessions):
        
        # if (s['Name']!='Uma') | (s['Date']!='2023-12-18'):
        #     continue
        datum = {}
        fullPath = os.path.join(saveDir, s['Name'], s['Date'])
        if (os.path.exists(fullPath)):
            for key in analysisList.keys():
                
                requiredFile = os.path.join(fullPath, analysisList[key])
                try:
                    if os.path.exists(requiredFile):
                        datum[key] = np.load(requiredFile)
                    else:
                        
                        if ('respP' in datum.keys()):
                            datum[key] = np.squeeze(np.ones((len(datum['respP']),*defaultShapes[key]))*np.nan)
                            
                        print(
                            f"for path {s}\n the file {analysisList[key]} did not exist")
                except:
                    print(
                        f"for path {s}\n could not load the file {analysisList[key]} ")
        if len(datum) > 0:
            datum['Id'] = np.repeat(s['Name'], len(datum["respP"]))
            datum['Date'] = np.repeat(s['Date'], len(datum["respP"]))
            datum['Nid'] = np.arange(len(datum["respP"]))
        
        
                        
        analysisData.append(datum)
    analysisData = pd.DataFrame(analysisData)
    
    return analysisData

def load_grating_pupil_fitting_data(saveDir,sessionCsv):
    sessions = pd.read_csv(sessionCsv)

    sessions = sessions.to_dict('records')
    
    
    analysisList = {'paramsOri': 'gratingOriTuning.stationay.params.npy', 'paramsOriSplit': "gratingOriTuning.paramsPupilStationary.npy", "varOriC": "gratingOriTuning.expVar.stationay.constant.npy", "varOriS": "gratingOriTuning.expVar.pupilStationarySplit.npy", "varOriN": "gratingOriTuning.expVar.stationay.noSplit.npy", "pvalOri": "gratingOriTuning.pVal.pupilStationarySplit.npy", "varOriSpecific": "gratingOriTuning.expVar.pupilStationarySplitSpecific.npy",
                         'paramsTf': 'gratingTfTuning.stationay.params.npy', 'paramsTfSplit': "gratingTfTuning.paramsPupilStationary.npy", "varTfC": "gratingTfTuning.expVar.stationay.constant.npy", "varTfS": "gratingTfTuning.expVar.pupilStationarySplit.npy", "varTfN": "gratingTfTuning.expVar.stationay.noSplit.npy", "pvalTf": "gratingTfTuning.pVal.pupilStationarySplit.npy", "varTfSpecific": "gratingTfTuning.expVar.pupilStationarySplitSpecific.npy",
                         'paramsSf': 'gratingSfTuning.stationay.params.npy', 'paramsSfSplit': "gratingSfTuning.paramsPupilStationary.npy", "varSfC": "gratingSfTuning.expVar.stationay.constant.npy", "varSfS": "gratingSfTuning.expVar.pupilStationarySplit.npy", "varSfN": "gratingSfTuning.expVar.stationay.noSplit.npy", "pvalSf": "gratingSfTuning.pVal.pupilStationarySplit.npy", "varSfSpecific": "gratingSfTuning.expVar.pupilStationarySplitSpecific.npy",
                         'paramsContrast': 'gratingContrastTuning.stationay.params.npy', 'paramsContrastSplit': "gratingContrastTuning.paramsPupilStationary.npy", "varContrastC": "gratingContrastTuning.expVar.stationay.constant.npy", "varContrastS": "gratingContrastTuning.expVar.pupilStationarySplit.npy", "varContrastN": "gratingContrastTuning.expVar.stationay.noSplit.npy", "pvalContrast": "gratingContrastTuning.pVal.pupilStationarySplit.npy", "varContrastSpecific": "gratingContrastTuning.expVar.pupilStationarySplitSpecific.npy",
                         "respP": "gratingResp.pVal.npy", "respDirection": "gratingResp.direction.npy"}

    defaultShapes = {"respP": (1,), "respDirection": (1,),
                     'paramsOri': (5,), 'paramsOriSplit': (5,2,), "varOriC": (1,), "varOriS": (1,), "varOriN": (1,), "pvalOri": (1,), "varOriSpecific": (2,), "nullOri": (5,2,500,),"permOri": (5,),
                    'paramsTf': (4,), 'paramsTfSplit': (4,2,), "varTfC":(1,), "varTfS": (1,), "varTfN": (1,), "pvalTf": (1,), "varTfSpecific": (4,),"permTf": (4,),  "nullTf": (4,2,500,),
                    'paramsSf': (4,), 'paramsSfSplit': (4,2,), "varSfC": (1,), "varSfS": (1,), "varSfN": (1,), "pvalSf": (1,), "varSfSpecific": (4,),"permSf": (4,),"nullSf": (4,2,500,),
                    'paramsContrast': (4,), 'paramsContrastSplit': (4,2,), "varContrastC": (1,), "varContrastS": (1,), "varContrastN": (1,), "pvalContrast": (1,), "varContrastSpecific": (4,),"permContrast": (4,),"nullContrast": (4,2,1000,),
                        }
    analysisData = []
    for si, s in enumerate(sessions):
        
        # if (s['Name']!='Uma') | (s['Date']!='2023-12-18'):
        #     continue
        datum = {}
        fullPath = os.path.join(saveDir, s['Name'], s['Date'])
        if (os.path.exists(fullPath)):
            for key in analysisList.keys():
                
                requiredFile = os.path.join(fullPath, analysisList[key])
                try:
                    if os.path.exists(requiredFile):
                        datum[key] = np.load(requiredFile)
                    else:
                        
                        if ('respP' in datum.keys()):
                            datum[key] = np.squeeze(np.ones((len(datum['respP']),*defaultShapes[key]))*np.nan)
                            
                        print(
                            f"for path {s}\n the file {analysisList[key]} did not exist")
                except:
                    print(
                        f"for path {s}\n could not load the file {analysisList[key]} ")
        if len(datum) > 0:
            datum['Id'] = np.repeat(s['Name'], len(datum["respP"]))
            datum['Date'] = np.repeat(s['Date'], len(datum["respP"]))
            datum['Nid'] = np.arange(len(datum["respP"]))
        
        
                        
        analysisData.append(datum)
    analysisData = pd.DataFrame(analysisData)
    
    return analysisData

def load_circle_analysis_data(saveDir,sessionCsv):
    # "D:\\Datadump\\Circles - Copy"
    
    sessions = pd.read_csv(sessionCsv)

    sessions = sessions.to_dict('records')
    
    
    circleAnalysisList = {'fitTimes': "circlesResp.bestTime.npy", 'responseValue':'circlesResp.max.npy','responsePval': "circlesResp.pVal.npy", 'sizeEv': "circlesSizeTuning.expVar.gamma.npy", 'sizeEvConst': "circlesSizeTuning.expVar.constant.npy",
                          'sizeProps': "circlesSizeTuning.params.npy", 'sizePrefSize': "circlesSizeTuning.prefSize.npy", 'sizeWidth': "circlesSizeTuning.width.npy", 'gaussCorr': "circlesRF.corr.npy",
                          'gaussPval': "circlesRF.pVal.npy", 'gaussEv': "circlesRF.expVar.gauss.npy", 'gaussEvConst': "circlesRF.expVar.constant.npy", 'gaussProps': "circlesRF.params.npy", "diameterMaps": "circlesRF.mapsDiameters.npy",
                          'max':'circlesResp.max.npy','maxOpp':'circlesResp.maxOpp.npy'
                          }
    
    circleAnalysisData = []
    
    for si, s in enumerate(sessions):
        datum = {}
        save = True
        fullPath = os.path.join(saveDir, s['Name'], s['Date'])
        if (os.path.exists(fullPath)):
            for key in circleAnalysisList.keys():
                requiredFile = os.path.join(fullPath, circleAnalysisList[key])
                if os.path.exists(requiredFile):
                    datum[key] = np.load(requiredFile)
                else:
                    print(
                        f"for path {s}\n the file {circleAnalysisList[key]} did not exist")
                    save = False
        if len(datum) > 0:
            datum['Id'] = np.repeat(s['Name'], len(datum["responsePval"]))
            datum['Date'] = np.repeat(s['Date'], len(datum["responsePval"]))
            datum['Nid'] = np.arange(len(datum["responsePval"]))
    
        circleAnalysisData.append(datum)
        
    circleAnalysisData = pd.DataFrame(circleAnalysisData)
    
    return circleAnalysisData

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

def create_criteria_shuffle(nullDist,paramsSplit,paramType='frequency'):
    
    nullDist = np.round(nullDist,4)
    paramsSplit = np.round(paramsSplit,4)
    if (paramType == 'frequency'):
        
        MInull = ((nullDist[:,0,1,:]+nullDist[:,1,1,:])-(nullDist[:,0,0,:]+nullDist[:,1,0,:]))/(0.5*(nullDist[:,0,1,:]+nullDist[:,1,1,:])+(nullDist[:,0,0,:]+nullDist[:,1,0,:]))
        dfNull = np.log2(nullDist[:, 2, 1,:])-np.log2(nullDist[:, 2, 0,:])
        sigNull = nullDist[:, 3, 1,:]-nullDist[:, 3, 0,:]
    
        aAmp = np.abs(np.sum(paramsSplit[:, [0, 1], 1], 1))
        qAmp = np.abs(np.sum(paramsSplit[:, [0, 1], 0], 1))
    
    
        dgainTf = (aAmp - qAmp)/(0.5*(aAmp + qAmp))
    
        fa = np.log2(paramsSplit[:, 2, 1])
        fq = np.log2(paramsSplit[:, 2, 0])
    
        df = fa-fq
    
        sigRatio = paramsSplit[:, 3, 1]-paramsSplit[:, 3, 0]
    
        pvalsMI = np.zeros(len(dgainTf))
        pvalsFreq = np.zeros(len(dgainTf))
        pvalsSig = np.zeros(len(dgainTf))
    
        for i in range(len(dgainTf)):
            score = sp.stats.percentileofscore(MInull[i,:], dgainTf[i])
            score = 100-score if score>50 else score
            score/=100
            score*=2
            pvalsMI[i] = score
            
            score = sp.stats.percentileofscore(dfNull[i,:], df[i])
            score = 100-score if score>50 else score
            score/=100
            score*=2
            pvalsFreq[i] = score
            
            score = sp.stats.percentileofscore(sigNull[i,:], sigRatio[i])
            score = 100-score if score>50 else score
            score/=100
            score*=2
            pvalsSig[i] = score
        
        MIVarPass = pvalsMI<0.05
        FreqVarPass = pvalsFreq<0.05
        SigmaVarPass = pvalsSig<0.05
        return MIVarPass, FreqVarPass, SigmaVarPass
    if (paramType == 'contrast'):        
        MInull = (nullDist[:,0,1,:]-nullDist[:,0,0,:])/(0.5*(nullDist[:,0,1,:]+nullDist[:,0,0,:]))
        c50Null = np.log2(nullDist[:, 1, 1,:])-np.log2(nullDist[:, 1, 0,:])
        cmaxNull = np.log2(nullDist[:, 2, 1,:])-np.log2(nullDist[:, 2, 0,:])
        csNull = np.log2(nullDist[:, 3, 1,:])-np.log2(nullDist[:, 3, 0,:])
       
        
        pvalsMI = np.zeros(len(paramsSplit))
        pvalsc50 = np.zeros(len(paramsSplit))
        pvalscmax = np.zeros(len(paramsSplit))
        pvalscs = np.zeros(len(paramsSplit))
        
        mi = (paramsSplit[:,0,1]-paramsSplit[:,0,0])/(0.5*(paramsSplit[:,0,1]+paramsSplit[:,0,0]))    
        c50 = np.log2(paramsSplit[:, 1, 1])-np.log2(paramsSplit[:, 1, 0])
        cmax = np.log2(paramsSplit[:, 2, 1])-np.log2(paramsSplit[:, 2, 0])
        cs = np.log2(paramsSplit[:, 3, 1])-np.log2(paramsSplit[:, 3, 0])
        for i in range(len(paramsSplit)):
            
            score = sp.stats.percentileofscore(MInull[i,:], mi[i])
            score = 100-score if score>50 else score
            score/=100
            score*=2
            pvalsMI[i] = score
            
            score = sp.stats.percentileofscore(c50Null[i,:], c50[i])
            score = 100-score if score>50 else score
            score/=100
            score*=2
            pvalsc50[i] = score
            
            score = sp.stats.percentileofscore(cmaxNull[i,:], cmax[i])
            score = 100-score if score>50 else score
            score/=100
            score*=2
            pvalscmax[i] = score
            
            score = sp.stats.percentileofscore(csNull[i,:], cs[i])
            score = 100-score if score>50 else score
            score/=100
            score*=2
            pvalscs[i] = score
            
        MIVarPass = pvalsMI<0.05
        c50VarPass = pvalsc50<0.05
        cmaxVarPass = pvalscmax<0.05
        csVarPass = pvalscs<0.05
        return MIVarPass, c50VarPass, cmaxVarPass,csVarPass
        
            
        
        
        


def get_contrast_params(p):  
    '''
    
    Parameters
    ----------
    p : parameters (must be of the format [4] or [NX4] or [NX4X2] or [NX4X2XShuffleN]
    
    Returns
    -------
    None.
    
    '''
    params = np.zeros_like(p)*np.nan
    
    # non-split parameter list 
    if (p.ndim==2):        
        noNanInd  = np.where(~np.all(np.isnan(p),axis=(1)))[0]
        for i in range(p.shape[0]):            
            params[i,:] = get_contrast_params(p[i,:])
        return params
    
    # split parameter list 
    if (p.ndim==3):
        noNanInd  = np.where(~np.all(np.isnan(p),axis=(1,2)))[0]
        for i in range(p.shape[0]):
            for j in range(p.shape[2]):
                params[i,:,j] = get_contrast_params(p[i,:,j])
        return params
    
    # split shuffle
    if (p.ndim==4):
        # save time by iterating only through valid neurons (not nans)
        noNanInd  = np.where(~np.all(np.isnan(p),axis=(1,2,3)))[0]
        for i in noNanInd:            
            for j in range(p.shape[2]):
                for k in range(p.shape[-1]):
                    params[i,:,j,k] = get_contrast_params(p[i,:,j,k])
        return params
        
    
    if (np.all(np.isnan(p))):
        return np.nan,np.nan,np.nan,np.nan
    
    
    
    tuner = ContrastTuner('contrast_modified')
    crange = np.arange(0,1.01,0.01)
    curve = tuner.func(crange,*p)
    
    # ignore flat curves
    if (np.nanmax(curve)==np.nanmin(curve)):
        return np.nan,np.nan,np.nan,np.nan
    
    
    if (p[0]<0):
        curve = -curve
    
    
    Rc = np.nanmax(curve)
    cMax = crange[np.argmax(curve)]    
    c50c = crange[crange<cMax][np.argmin(np.abs(curve[crange<cMax]-Rc/2))]
    
    # normalise to get a standardized slop
    curve_ = curve-np.nanmin(curve)
    curve_ = curve_/np.nanmax(curve_)
    curveDiff = np.diff(curve,prepend=True,axis=0)/0.01    
    c50s = ((curveDiff[crange==c50c][0]))
    
    params = np.array([Rc,c50c,cMax,c50s])
    return params
       
def get_histogram_bins(searchMatrix, bins, valCol=0, rel=False, returnCounts=False,customCounts = None):
    '''
    

    Parameters
    ----------
    searchMatrix : TYPE
        DESCRIPTION.
    bins : TYPE
        DESCRIPTION.
    valCol : TYPE, optional
        DESCRIPTION. The default is 0.
    rel : TYPE, optional
        DESCRIPTION. The default is False.
    returnCounts : TYPE, optional
        DESCRIPTION. The default is False.
    customCounts : TYPE
        cases when one to check actual percentage e.g. only neurons that could be measured at all.
        has to be the size of the other columns
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    
    vals = np.zeros((searchMatrix.shape[1], len(bins)))
    counts = np.zeros((len(bins)))
    
    if (not customCounts is None):
        sizeDiff = searchMatrix.shape[-1] - customCounts.shape[-1]
        if (sizeDiff>0):
            customCounts = np.hstack((np.ones((searchMatrix.shape[0],sizeDiff)),customCounts))
        

    for bi in range(len(bins)-1):
        bInds = np.where((searchMatrix[:, valCol] >= bins[bi]) & (
            searchMatrix[:, valCol] < bins[bi+1]))[0]
        vals[:, bi] = np.sum(searchMatrix[bInds, :], 0)
        if (rel):
            count = len(bInds)
            if (not customCounts is None):
                count = np.sum(customCounts[bInds],0)
            vals[:, bi] = vals[:, bi]/count
            counts[bi] = len(bInds)
    bInds = np.where(searchMatrix[:, valCol] >= bins[bi+1])[0]
    vals[:, bi+1] = np.sum(searchMatrix[bInds, :], 0)
    if (rel):
        vals[:, bi+1] = vals[:, bi+1]/len(bInds)
    if (returnCounts):
        return vals, counts
    return vals