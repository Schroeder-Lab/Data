# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:57:30 2023

@author: Liad
"""
import numpy as np
from matplotlib import pyplot as plt
import random
import sklearn
import scipy as sp

from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd

import os
import glob
import re


def get_calcium_aligned(signal, time, eventTimes, window, planes, delays, baseline_sub=True, blTime=0.5):
    aligned = []
    run = 0
    ps = np.unique(planes).astype(int)

    for p in range(len(ps)):
        aligned_tmp, t = align_stim(
            signal[:, np.where(planes == ps[p])[0]],
            time + delays[0, ps[p]],
            eventTimes,
            window,
        )
        if run == 0:
            aligned = aligned_tmp
            run += 1
        else:
            aligned = np.concatenate((aligned, aligned_tmp), axis=2)
    if (baseline_sub):
        aligned = np.array(aligned)
        aligned -= np.nanmean(aligned[(t < 0) & (t >= -blTime), :, :], 0)
    return np.array(aligned), t


def align_stim(signal, time, eventTimes, window, timeUnit=1, timeLimit=1):
    aligned = []
    t = []

    dt = np.nanmedian(np.diff(time, axis=0))
    if timeUnit == 1:
        w = np.rint(window / dt).astype(int)
    else:
        w = window.astype(int)
    maxDur = signal.shape[0]
    if window.shape[0] == 1:  # constant window
        mini = np.min(w[:, 0])
        maxi = np.max(w[:, 1])
        tmp = np.array(range(mini, maxi))
        w = np.tile(w, ((eventTimes.shape[0], 1)))
    else:
        if window.shape[0] != eventTimes.shape[0]:
            print("No. events and windows have to be the same!")
            return
        else:
            mini = np.min(w[:, 0])
            maxi = np.max(w[:, 1])
            tmp = range(mini, maxi)
    t = tmp * dt

    aligned = np.zeros(
        (t.shape[0], eventTimes.shape[0], signal.shape[1]))*np.nan

    for ev in range(eventTimes.shape[0]):
        #     evInd = find(time > eventTimes(ev), 1);

        wst = w[ev, 0]
        wet = w[ev, 1]

        evInd = np.where(time >= eventTimes[ev])[0]
        if len(evInd) == 0:
            continue
        else:
            # None
            # if dist is bigger than one second stop
            if np.any((time[evInd[0]] - eventTimes[ev]) > timeLimit):
                continue
        st = evInd[0] + wst  # get start
        et = evInd[0] + wet  # get end

        alignRange = np.array(
            range(
                np.where(tmp == wst)[0][0], np.where(tmp == wet - 1)[0][0] + 1
            )
        )

        sigRange = np.array(range(st, et))

        valid = np.where((sigRange >= 0) & (sigRange < maxDur))[0]

        aligned[alignRange[valid], ev, :] = signal[sigRange[valid], :]
    return aligned, t
