import numpy as np
from matplotlib import pyplot as plt
import csv
import glob
import re
from numba import jit, cuda
import numba
import pandas as pd
import scipy as sp
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os
from Data.TwoP.general import *


"""Pre-process data recorded with Bonsai."""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:26:53 2022

@author: Liad J. Baruchin
"""


def get_nidaq_channels(niDaqFilePath, numChannels=None, plot=False):
    """
    Gets the nidaq channels.

    Parameters
    ----------
    niDaqFilePath : string
        the path of the nidaq file.
    numChannels : int, optional
        Number of channels in the file, if none will look for a file describing
        the channels. The default is None.

    Returns
    -------
    niDaq : np.ndarray
        The matrix of the niDaq signals [time X channels].
    nidaqTime: array [s]
        The clock time of each nidaq timepoint.

    """
    # Gets the number of channels from the nidaq channels csv file which is
    # automatically generated from the Bonsai script.
    if numChannels is None:
        dirs = glob.glob(os.path.join(niDaqFilePath, "nidaqChannels*.csv"))
        if len(dirs) == 0:
            print("ERROR: no channel file and no channel number given")
            return None
        channels = np.loadtxt(dirs[0], delimiter=",", dtype=str)
        if len(channels.shape) > 0:
            numChannels = len(channels)
            nidaqSignals = dict.fromkeys(channels, None)
        else:
            numChannels = 1
            nidaqSignals = {str(channels): None}
    else:
        channels = range(numChannels)
    # Gets the actual nidaq file and extract the data from it.
    niDaqFilePath = get_file_in_directory(niDaqFilePath, "NidaqInput")
    niDaq = np.fromfile(niDaqFilePath, dtype=np.float64)
    if int(len(niDaq) % numChannels) == 0:
        niDaq = np.reshape(niDaq, (int(len(niDaq) / numChannels), numChannels))
    else:
        # File was somehow screwed. Finds the good bit of the data.
        correctDuration = int(len(niDaq) // numChannels)
        lastGoodEntry = correctDuration * numChannels
        niDaq = np.reshape(
            niDaq[:lastGoodEntry], (correctDuration, numChannels)
        )
    # Option to plot the channels.
    if plot:
        f, ax = plt.subplots(max(2, numChannels), sharex=True)
        for i in range(numChannels):
            ax[i].plot(niDaq[:, i])
    nidaqTime = np.arange(niDaq.shape[0]) / 1000

    return niDaq, channels, nidaqTime


def assign_frame_time(frameClock, th=0.5, fs=1000, plot=False):
    """
    Assigns a time in s to a frame time.

    Parameters
    ----------
    frameClock : np.array[frames]
        The signal of the frame clock from the nidaq.
    th : float, optional
        The threshold for the tick peaks.
        The default is 0.5.
    fs : float, optional
        The frame rate of acquisition. The default is 1000.
    plot : plt plot, optional
        Plot to inspect. The default is False.

    Returns
    -------
    np.array[frames]
        Frame start times (s).

    """
    # Frame times
    # pkTimes,_ = sp.signal.find_peaks(-frameClock,threshold=th)
    # pkTimes = np.where(frameClock<th)[0]
    # fdif = np.diff(pkTimes)
    # longFrame = np.where(fdif==1)[0]
    # pkTimes = np.delete(pkTimes,longFrame)
    # recordingTimes = np.arange(0,len(frameClock),0.001)
    # frameTimes = recordingTimes[pkTimes]
    # threshold = 0.5

    # Gets the timepoints where the frame clock signal is above a certain
    # threshold.
    pkTimes = np.where(np.diff(frameClock > th, prepend=False, axis=0))[0]
    # pkTimes = np.where(np.diff(np.array(frameClock > 0).astype(int),prepend=False)>0)[0]
    # Option to plot the frame clock and the times.
    if plot:
        f, ax = plt.subplots(1)
        ax.plot(frameClock)
        ax.plot(pkTimes, np.ones(len(pkTimes)) * np.min(frameClock), "r*")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Amplitude (V)")
    # Returns the frame start times in s.
    return pkTimes[::2] / fs


def detect_photodiode_changes(
    photodiode,
    plot=False,
    kernel=10,
    upThreshold=0.2,
    downThreshold=0.4,
    fs=1000,
    waitTime=10000,
):

    """
    The function detects photodiode changes using a 'Schmitt Trigger', that is,
    by detecting the signal going up at an earlier point than the signal going
    down,the signal is filtered and smoothed to prevent nosiy bursts distorting
    the detection.

    Parameters
    ----------
    photodiode : np.array[frames]
        The signal of the photodiode from the nidaq.
    kernel : int
        The kernel for median filtering, The default is 10.
    upThreshold : float
        The lower limit for the photodiode signal. The default is 0.2.
    downThreshold : float
        The upper limit for the photodiode signal. The default is 0.4.
    fs: int
        The frequency of acquisiton. The default is 1000.
    plot: plt plot
        Plot to inspect. The default is False.
    waitTime: float
        The delay time until protocol start. The default is 5000.

    Returns
    -------
    np.array[frames]
        Photoiode changes (s); up to the user to decide what on and off mean.
    """

    # b,a = sp.signal.butter(1, lowPass, btype='low', fs=fs)

    sigFilt = photodiode.copy()
    # sigFilt = sp.signal.filtfilt(ba,photodiode)
    # Creates the convolving window of the kernel size specified.
    w = np.ones(kernel) / kernel
    # sigFilt = sp.signal.medfilt(sigFilt,kernel)
    # Smoothes the photodiode signal.
    sigFilt = np.convolve(sigFilt[:, 0], w, mode="same")
    sigFilt_raw = sigFilt.copy()
    sigFilt_diff = np.diff(sigFilt)  # TODO: NOT used, remove.

    # Gets the max and min signal amplitude.
    maxSig = np.max(sigFilt)
    minSig = np.min(sigFilt)

    # Gets the mean and standard deviation values during the wait time.
    mean_waitTime = np.nanmean(sigFilt[:waitTime])
    std_waitTime = np.nanstd(sigFilt[:waitTime])

    # Sets the upper and lower threshold.
    thresholdU = (maxSig - minSig) * upThreshold
    thresholdD = (maxSig - minSig) * downThreshold
    threshold = (maxSig - minSig) * 0.5

    # Finds threshold crossings. #TODO: remove unused variables.
    uBaselineCond = sigFilt > (mean_waitTime + 1 * std_waitTime)
    uThresholdCond = sigFilt > thresholdU
    dBaselineCond = sigFilt > (mean_waitTime + 1 * std_waitTime)
    dThresholdCond = sigFilt > thresholdD
    crossingsU = np.where(
        np.diff(np.array(uThresholdCond).astype(int), prepend=False) > 0
    )[0]
    crossingsD = np.where(
        np.diff(np.array(dThresholdCond).astype(int), prepend=False) < 0
    )[0]
    crossingsU = np.delete(crossingsU, np.where(crossingsU < waitTime)[0])
    crossingsD = np.delete(crossingsD, np.where(crossingsD < waitTime)[0])
    crossings = np.sort(np.unique(np.hstack((crossingsU, crossingsD))))

    # For the first crosssing might be an issue detecting change if the
    # entire baseline is over threshold. Looks for the first that is over it
    # or under it, and add them if they appear before the first detected change.
    changeInd = np.where(sigFilt > mean_waitTime + std_waitTime)[0]
    changeInd = changeInd[changeInd >= waitTime]
    if (
        (len(changeInd) > 0)
        and (changeInd[0] < crossings[0])
        and (crossingsD[0] < crossingsU[0])
    ):
        crossings = np.append(changeInd[0], crossings)

    # changeInd = np.where(sigFilt < thresholdD)[0]
    # changeInd = changeInd[changeInd >= waitTime]
    # if (
    #     (len(changeInd) > 0)
    #     and (changeInd[0] < crossings[0])
    #     and (crossingsD[0] > crossingsU[0])
    # ):
    #     crossings = np.append(changeInd[0], crossings)
    # Option to plot the photodiode signal withe the detected changes.
    if plot:
        f, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(photodiode, label="photodiode raw")
        ax.plot(sigFilt_raw, label="photodiode filtered")
        ax.plot(crossings, np.ones(len(crossings)) * threshold, "g*")
        ax.hlines([thresholdU], 0, len(photodiode), "k")
        ax.hlines([thresholdD], 0, len(photodiode), "k")
        # ax.plot(st,np.ones(len(crossingsD))*threshold,'r*')
        ax.legend()
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Amplitude (V)")
    return crossings / fs


def detect_wheel_move(
    moveA, moveB, timestamps, rev_res=1024, total_track=59.847, plot=False
):
    """
    Converts the rotary encoder data to velocity and distance travelled.
    At the moment uses only moveA.

    Parameters
    ----------
    moveA : np.array[frames]
        The first channel of the rotary encoder.
    moveB : np.array[frames]
        The second channel of the rotary encoder..
    timestamps : np.array[frames]
        The timestamps associated with the frames.
    rev_res : int, optional
        The rotary encoder resoution. The default is 1024.
    total_track : TYPE, optional
        The total length of the track. The default is 59.847.
    plot : plt plot
        Plot to inspect. The default is False.

    Returns
    -------
    velocity : np.array[frames]
        Velocity[cm/s].
    distance : np.array[frames]
        Distance travelled [cm].

    """

    moveA = np.round(moveA / np.max(moveA)).astype(bool)
    moveB = np.round(moveB / np.max(moveB)).astype(bool)
    counterA = np.zeros(len(moveA))
    counterB = np.zeros(len(moveB))

    # Detects A move.
    risingEdgeA = np.where(np.diff(moveA > 0, prepend=True))[0]
    risingEdgeA = risingEdgeA[moveA[risingEdgeA] == 1]
    risingEdgeA_B = moveB[risingEdgeA]
    counterA[risingEdgeA[risingEdgeA_B == 0]] = 1
    counterA[risingEdgeA[risingEdgeA_B == 1]] = -1

    # Detects B move.
    risingEdgeB = np.where(np.diff(moveB > 0, prepend=True))[
        0
    ]  # np.diff(moveB)

    risingEdgeB = risingEdgeB[moveB[risingEdgeB] == 1]
    risingEdgeB_A = moveB[risingEdgeB]
    counterA[risingEdgeB[risingEdgeB_A == 0]] = -1
    counterA[risingEdgeB[risingEdgeB_A == 1]] = 1

    # Gets how much one move means in distance travelled.

    dist_per_move = total_track / rev_res
    # Gets th distance throughout the whole experiment.
    instDist = counterA * dist_per_move
    distance = np.cumsum(instDist)
    # Prepares the windows used for converting the distance and counting the time.
    averagingTime = int(np.round(1 / np.median(np.diff(timestamps))))
    sumKernel = np.ones(averagingTime)
    tsKernel = np.zeros(averagingTime)
    tsKernel[0] = 1
    tsKernel[-1] = -1

    # Taking the difference and
    distDiff = np.diff(distance)
    velocity = (
        sp.ndimage.gaussian_filter1d(distDiff, averagingTime / 2)
        * averagingTime
    )

    # if (plot):
    #     f,ax = plt.subplots(3,1,sharex=True)
    #     ax[0].plot(moveA)
    #     # ax.plot(np.abs(ADiff))
    #     ax[0].plot(Ast,np.ones(len(Ast)),'k*')
    #     ax[0].plot(Aet,np.ones(len(Aet)),'r*')
    #     ax[0].set_xlabel('time (ms)')
    #     ax[0].set_ylabel('Amplitude (V)')

    #     ax[1].plot(distance)
    #     ax[1].set_xlabel('time (ms)')
    #     ax[1].set_ylabel('distance (mm)')

    #     ax[2].plot(track)
    #     ax[2].set_xlabel('time (ms)')
    #     ax[2].set_ylabel('Move')

    # movFirst = Amoves>Bmoves

    return velocity, distance


def get_sparse_noise(filePath, size=None):

    """
    Pulls the sparse noise from the directory.

    Parameters
    ----------
    filePath : str
        The full file path for the sparse noise file.
    size: tuple
        A tuple for the size of the screen (into how many squares the screen
        is divided into). The default is None.

    Returns
    -------
    np.array [frames X size[0] X size[1]]
        The sparse map.
    """

    # Loads sparse noise binary file.
    filePath_ = get_file_in_directory(filePath, "sparse")
    sparse = np.fromfile(filePath_, dtype=np.dtype("b")).astype(float)

    if size is None:
        # Gets experimental details (size of the screen) from the props file.
        dirs = glob.glob(os.path.join(filePath, "props*.csv"))
        if len(dirs) == 0:
            print("ERROR: no props file given")
            return None
        # Gets the size of the squares from the props.
        size = np.loadtxt(dirs[0], delimiter=",", dtype=int)
    # Reassigns values in the sparse array.
    sparse[sparse == -128] = 0.5
    sparse[sparse == -1] = 1
    # Reshapes the sparse array to represent the size of the screen and where
    # within this grid the black or white squares appeared.
    sparse = np.reshape(
        sparse, (int(len(sparse) / (size[1] * size[0])), size[0], size[1])
    )
    # Rearranges the sparse map.
    return np.moveaxis(np.flip(sparse, 2), -1, 1)


def get_log_entry(filePath, entryString):
    """


    Parameters
    ----------
    filePath : str
        the path of the log file.
    entryString : the string of the entry to look for

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the
        props and their values.

    """
    rowN = 0
    StimProperties = []
    exactLogPath = glob.glob(os.path.join(filePath, "Log*.csv"))
    if len(exactLogPath) == 0:
        return None
    else:
        filePath = exactLogPath[0]      
    
    with open(filePath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", quotechar="|")

        for row in reader:
            a = []
            if "Video" in row[0]:
                None
            for p in range(len(entryString)):
                # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
                fullRow = "".join(row)
                m = re.findall(
                    entryString[p], fullRow
                )  # row[np.min([len(row) - 1, p])

                if len(m) > 0:
                    # a.append(str(rowN) + "," + row[np.min([len(row) - 1, p])])

                    # if len(a) > 0:
                    stimProps = {entryString[p]: str(rowN) + "," + fullRow}
                    # stimProps[entryString[p]] = a[p]
                    StimProperties.append(stimProps)
            rowN += 1
    return StimProperties


def get_stimulus_info(filePath, props=None):
    """


    Parameters
    ----------
    filePath : str
        the path of the log file.
    props : np.array of str
        the names of the properties to extract, if None looks for a file.
        The default is None.

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the
        props and their values.

    """
    # Gets the experimental details from the props file
    if props is None:
        dirs = glob.glob(os.path.join(filePath, "props*.csv"))
        if len(dirs) == 0:
            print("ERROR: no props file given")
            return None

        props = np.loadtxt(dirs[0], delimiter=",", dtype=str)
    # Gets the log file which contains all the parameters for each stimulus
    # presentation.
    logPath = glob.glob(os.path.join(filePath, "Log*"))
    if len(logPath) == 0:
        return None
    logPath = logPath[
        0
    ]  # Gets the first log file in case there's more than 1.
    # Creates a dictionary for the stimulus properties.
    StimProperties = {}
    # for p in range(len(props)):
    #     StimProperties[props[p]] = []

    searchTerm = ""
    # Finds the different parameters defined in the props file.
    for p in range(len(props)):
        searchTerm += props[p] + "=([a-zA-Z0-9_.-]*)"

        if p < len(props) - 1:
            searchTerm += "|"
    # Reads the log csv file.
    with open(logPath, newline="") as csvfile:
        allLog = csvfile.read()
    # Gets the values for each stimulus repetition for all the parameters.
    for p in range(len(props)):
        m = re.findall(props[p] + "=([.-a-zA-Z0-9_\\\\]*)", allLog)
        # Appends the list of each parameter into a dictionary.
        if len(m) > 0:
            StimProperties[props[p]] = m
    #         # #         stimProps[props[p]] = a[p]
    #         # #     StimProperties.append(stimProps)

    # with open(logPath, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in reader:
    #         # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
    #         m = re.findall(searchTerm, str(row))
    #         if (len(m)>0):
    #             StimProperties.append(m)
    #         # a = []
    #         # for p in range(len(props)):
    #         #     # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
    #         #     m = re.findall(props[p]+'=([a-zA-Z0-9_.-]*)', row[np.min([len(row)-1,p])])
    #         #     if (len(m)>0):
    #         #         # a.append(m[0])
    #         #         StimProperties[props[p]].append(m[0])
    #         # # if (len(a)>0):
    #         # #     stimProps = {}
    #         # #     for p in range(len(props)):
    #         # #         stimProps[props[p]] = a[p]
    #         # #     StimProperties.append(stimProps)
    # Returns a pandas dataframe of the stimulus properties dictionary.
    return pd.DataFrame(StimProperties)


# @jit(forceobj=True)
def get_arduino_data(arduinoDirectory, plot=False):
    """
    Retrieves the arduino data, regularises it (getting rid of small intervals)
    Always assumes last entry is the timepoints.

    Parameters
    ----------
    arduinoFilePath : str
        The path of the arduino file.
    plot : bool, optional
        Whether or not to plot all the channels. The default is False.

    Returns
    -------
    csvChannels : array-like [time X channels]
        All the channels recorded by the arduino.

    """
    # Gets the arduino file.
    arduinoFilePath = get_file_in_directory(arduinoDirectory, "ArduinoInput")
    # Loads the arduino data.
    csvChannels = np.loadtxt(arduinoFilePath, delimiter=",")
    # arduinoTime = csvChannels[:,-1]
    # Calculates the arduino time.
    # arduinoTime = np.arange(csvChannels.shape[0]) / 1000
    # arduinoTimeDiff = np.diff(arduinoTime,prepend=True)
    # normalTimeDiff = np.where(arduinoTimeDiff>-100)[0]
    # csvChannels = csvChannels[normalTimeDiff,:]
    # # convert time to second (always in ms)
    arduinoTime = csvChannels[:,-1]/1000

    # Starts arduino time at zero.
    arduinoTime -= arduinoTime[0]
    csvChannels = csvChannels[
        :, :-1
    ]  # Takes all channels except the timepoints.
    numChannels = csvChannels.shape[1]  # Gets number of channels.
    if plot:  # Option to plot all channels.
        f, ax = plt.subplots(numChannels, sharex=True)
        for i in range(numChannels):
            ax[i].plot(arduinoTime, csvChannels[:, i])
    dirs = glob.glob(os.path.join(arduinoDirectory, "arduinoChannels*.csv"))
    if len(dirs) == 0:
        channelNames = []
    else:  # Gets the names of each channel from the channel csv file.
        channelNames = np.loadtxt(dirs[0], delimiter=",", dtype=str)
    return csvChannels, channelNames, arduinoTime


# @jit((numba.b1, numba.b1, numba.double, numba.double,numba.int8))
def arduino_delay_compensation(
    nidaqSync, ardSync, niTimes, ardTimes, batchSize=100
):
    """
    Corrects the arduino signal time to be synched to the nidaq time. This is
    important given that different devices were used to acquire these signals
    and in order to ensure that the signals are aligned correctly, the signals
    need to be synched.

    Parameters
    ----------
    nidaqSync : array like[frames]
        The synchronisation signal from the nidaq or any non-arduino acquisiton
        system.
    ardSync : array like[frames]
        The synchronisation signal from the arduino.
    niTimes : array like [s]
        the timestamps of the acqusition signal.
    ardTimes : array ike [s]
        The timestamps of the arduino signal.
    batchSize : int
        The interval over which to sample. The default is 100.

    Returns
    -------
    newArdTimes : array like [s]
        The corrected arduino signal. Shifting the time either forward or
        backwards in relation to the faster acquisition.

    """
    niTick = np.round(nidaqSync).astype(bool)
    ardTick = np.round(ardSync).astype(bool)

    ardFreq = np.median(np.diff(ardTimes))
    # Gets where the ni sync signal changes.
    niChange = np.where(np.diff(niTick, prepend=True) > 0)[0][:]

    # Checks that the first state change is clear.
    if (niChange[0] == 0) or (niChange[1] - niChange[0] > 50):
        niChange = niChange[
            1:
        ]  # takes the second time point as the true start.
    # Gets the times for the changes.
    niChangeTime = niTimes[niChange]
    # Gets the how long each change lasts.
    niChangeDuration = np.round(np.diff(niChangeTime), 4)
    # Normalises the duration of the change.
    niChangeDuration_norm = (
        niChangeDuration - np.mean(niChangeDuration)
    ) / np.std(niChangeDuration)
    # Does the same as above for the arduino.
    ardChange = np.where(np.diff(ardTick, prepend=True) > 0)[0][:]
    # Checks that first state change is clear.
    if ardChange[0] == 0:
        ardChange = ardChange[1:]
    ardChangeTime = ardTimes[ardChange]
    ardChangeDuration = np.round(np.diff(ardChangeTime), 4)
    # niChangeTime = np.append(niChangeTime,np.zeros_like(ardChangeTime))
    ardChangeDuration_norm = (
        ardChangeDuration - np.mean(ardChangeDuration)
    ) / np.std(ardChangeDuration)

    newArdTimes = ardTimes.copy()
    # reg = linear_model.LinearRegression()

    mses = []
    mse_prev = 10**4  # TODO: remove unused variable.
    a_list = []
    b_list = []
    # a = []
    # b = []
    # passRange = min(batchSize,len(niChangeTime))#-len(ardChangeTime)
    passRange = 100  # len(niChangeTime)

    if passRange > 0:
        # Compares the lengths of the nidaq and the arduino. In case the length
        # between the two is different, performs linear regression and compares the
        # first 100 times to see which ni time matches best with the arduino time.
        for i in range(passRange):
            # y = niChangeTime[i:]
            # x = ardChangeTime[:len(y)]
            y = niChangeDuration_norm[i:]
            x = ardChangeDuration_norm[: len(y)]
            y = y - y[0]
            minTime = np.min([len(x), len(y)])
            lenDif = len(x) - len(y)
            x = x[:minTime]
            y = y[:minTime]

            if lenDif > 0:
                x = x[:-lenDif]
            a_, b_, mse = linear_analytical_solution(x, y)
            mses.append(mse)
            a_list.append(a_)
            b_list.append(b_)

            # stop when error starts to increase, to save time
            # if (mse>=mse_prev):
            #     break;
            # mse_prev = mse
        bestTime = np.argmin(mses[0:])
        # bestTime = i-1
        # Starts the ni time where it matches the arduino time most.
        niChangeTime = niChangeTime[bestTime:]
        # Gets the minimum length between the ni time and arduino time.
        minTime = np.min([len(niChangeTime), len(ardChangeTime)])
        maxOverlapTime = niChangeTime[minTime - 1]  # TODO: remove, not used.

        # Only takes the ni and arduino change times until the min time.
        niChangeTime = niChangeTime[:minTime]
        ardChangeTime = ardChangeTime[:minTime]

        # Gets the duration of each change (rounded to 4 decimal points).
        ardChangeDuration = np.round(np.diff(ardChangeTime), 4)
        niChangeDuration = np.round(np.diff(niChangeTime), 4)

        # Checks the difference between the two and if the duration of each
        # acquisition matches between them.
        a = niChangeTime[0] - ardChangeTime[0]
        b = np.median(niChangeDuration / ardChangeDuration)

        lastPoint = 0
        # Within this for loop, finds where there are misalignments due to
        # potentially uneven acquisition of the signal and realigns it.
        for i in range(0, len(ardChangeTime) + 1, batchSize):
            if i >= len(ardChangeTime):
                continue

            x = ardChangeTime[i : np.min([len(ardChangeTime), i + batchSize])]
            y = niChangeTime[i : np.min([len(ardChangeTime), i + batchSize])]

            a, b, mse = linear_analytical_solution(x, y)

            ind = np.where((newArdTimes >= lastPoint))[0]
            newArdTimes[ind] = b * newArdTimes[ind] + a

            ardChangeTime = ardChangeTime * b + a

            lastPoint = (
                ardChangeTime[np.min([len(ardChangeTime) - 1, i + batchSize])]
                + 0.00001
            )
    return newArdTimes


def get_piezo_trace_for_plane(
    piezo,
    frameTimes,
    piezoTime,
    imagingPlanes,
    selectedPlanes=None,
    vRatio=5 / 400,
    winSize=20,
    batchFactor=100,
):
    """
    Calculates the average movement of the piezo across z-axis in one frame for all planes.
    Location in depth (in microns) is for each milisecond within one plane.

    Parameters
    ----------
    piezo : np.array[nidaq timepoints]
        Piezo trace.
    frameTimes : np.array [frames]
        Frame start Times (s).
    piezoTime : np.array[nidaq timepoints]
        The time in seconds of each timepoint in the piezo trace.
    imagingPlanes : int
        Number of planes imaged.
    selectedPlanes : np.array[selectedPlanes], optional
        Certain selected planes if wanting to only get the data for specific planes. The default is None.
    vRatio : float, optional
        the range of voltage over the distance travelled in Z. The default is 5 / 400.
    winSize : int, optional
        the window size over which to smooth the trace. The default is 20.
    batchFactor : int, optional
        The number of frames to sample over. The default is sampling over every 100th frame.

    Returns
    -------
    planePiezo : np.array [miliseconds in one frame, nplanes]
        Movement of piezo across z-axis for all planes.
        Location in depth (in microns) is for each milisecond within one plane.

    """
    # Unless certain planes are chosen, all the imaging planes will be taken into account.
    if selectedPlanes is None:
        # Creates a range object of the range [no. of imaging planes].
        selectedPlanes = range(imagingPlanes)
    else:
        # Creates an array of at least 1D with the selected plane values.
        selectedPlanes = np.atleast_1d(selectedPlanes)
    # Returns a Hanning window of size winSize.
    w = np.hanning(winSize)
    # Divides the values in the window by the sum of the values.
    # This averages the window so that the area under the curve is 1.
    w /= np.sum(w)
    # Smoothes the piezo trace with the hanning window from above to remove irregularities in the trace.
    piezo = np.convolve(piezo, w, "same")

    # Subtracts the minimum value in the piezo trace from the piezo trace to obtain positive values only.
    piezo -= np.min(piezo)
    # Divides the piezo trace by the voltage ratio to convert the voltage values into distance in microns.
    piezo /= vRatio
    # Determines the duration of each frame in miliseconds.
    traceDuration = int(np.median(np.diff(frameTimes)) * 1000)  # convert to ms
    # Creates an array where the location in depth is for each milisecond within one plane.
    planePiezo = np.zeros((traceDuration, len(selectedPlanes)))

    # Runs over the imaging planes and calculates the average depth per frame every 100th frame.
    for i in range(len(selectedPlanes)):
        plane = selectedPlanes[i]

        # Below section takes an average of piezo trace for each plane, by sampling every 100th frame.

        # Determines the time at which the piezo starts and ends for each plane but ignoring the first frame
        # because the location of the first frame is when the piezo starts moving so it is inaccurate.
        piezoStarts = frameTimes[imagingPlanes + plane :: imagingPlanes]
        piezoEnds = frameTimes[imagingPlanes + plane + 1 :: imagingPlanes]
        # Determines the range over which to sample over the piezo trace given the batchFactor specified.
        piezoBatchRange = range(0, len(piezoStarts), batchFactor)
        # Creates the array for the piezo location for each milisecond in each batch.
        avgTrace = np.zeros((traceDuration, len(piezoBatchRange)))
        for avgInd, pi in enumerate(piezoBatchRange):
            # Determines the section of the piezo trace to take into account given the piezo start and end times
            # specified above.
            inds = np.where(
                (piezoTime >= piezoStarts[pi]) & (piezoTime < piezoEnds[pi])
            )
            # Gets the array for the piezo location for each milisecond in each batch.
            avgTrace[:, avgInd] = piezo[inds][: len(avgTrace[:, avgInd])]
        # Calculates the average piezo location for each milisecond in the frame.
        avgTrace = np.nanmean(avgTrace, 1)
        # Combines the average piezo location from each plane.
        planePiezo[:, i] = avgTrace
    return planePiezo


def adjustPiezoTrace():
    None


def get_file_in_directory(directory, simpleName):
    """
    Gets the file path of the first file with the same name.
    For example,if a directory contains two ArduinoInput files, such as
    ArduinoInput0.csv and ArduinoInput1.csv, it will return the file path of
    ArduinoInput0.csv.

    Parameters
    ----------
    directory : str
        The directory where the respective files are located.
    simpleName : str
        a keyword that describes the file. For example, for the arduino input
        file mentioned above, such a keyword would be "ArduinoInput".

    Returns
    -------
    file[0] : str
       The file path of the first file with the same name. .

    """
    file = glob.glob(os.path.join(directory, simpleName + "*"), recursive=True)
    if len(file) > 0:
        return file[0]
    else:
        return None


def get_piezo_data(ops):
    """
    Extracts all the data needed to run the above function, get_piezo_trace.
    This includes:
            - the current working directory
            - the number of planes
            - the nidaq channels, especially the frameclock, the niday times and the piezo data

    Parameters
    ----------
    ops : dict
        dictionary from the suite2p folder including all the input settings such as
        the number of planes.

    Returns
    -------
    planePiezo : np.array [miliseconds in one frame, nplanes]
        Movement of piezo across z-axis for all planes.
        Location in depth (in microns) is for each milisecond within one plane.


    """
    # Loads the current experiment for which to get the piezo data.
    piezoDir = ops["data_path"][0]
    # Loads the number of planes from the ops file.
    nplanes = ops["nplanes"]
    # Returns all the nidaq channels, the number of channels and the nidaq time.
    nidaq, channels, nt = get_nidaq_channels(piezoDir, plot=False)
    # Loads the frameclock from the nidaq.
    frameclock = nidaq[:, channels == "frameclock"]
    # Returns the time at which each frame was acquired.
    frames = assign_frame_time(frameclock, plot=False)
    # Loads the piezo.
    piezo = nidaq[:, channels == "piezo"].copy()[:, 0]
    # Returns the movement of the piezo across the z-axis for all planes.
    planePiezo = get_piezo_trace_for_plane(
        piezo, frames, nt, imagingPlanes=nplanes
    )
    return planePiezo


def get_recorded_video_times(di, searchTerms, cleanNames):
    """
    Gets the recorded video times from the log files that bonsai is saving

    Parameters
    ----------
    di : str
        The directory where the log file resides.
    searchTerms : list (str)
        The terms used to represent the different video recordings .Last has to be the Nidaq.
    cleanNames : str
        A nicer names to use when logging the entries.

    Returns
    -------
    None.

    """
    log = get_log_entry(di, searchTerms)
    log_df = pd.DataFrame(log)

    # create rename dict
    renameDict = {}
    for i in range(len(searchTerms)):
        renameDict[searchTerms[i]] = cleanNames[i]
    log_df.rename(
        columns=renameDict,
        inplace=True,
    )

    # change NI values
    occurenceInds = log_df[cleanNames[-1]].index[
        log_df[cleanNames[-1]].notna()
    ]
    a = log_df.loc[occurenceInds, cleanNames[-1]] = np.arange(
        len(occurenceInds)
    )

    # vidLogEye = log_df["EyeVid"].values
    # vidLogBody = log_df["BodyVid"].values
    niLog = log_df[cleanNames[-1]].dropna().values
    logFramesNi = np.zeros((len(niLog), 2)) * np.nan
    for j in range(len(niLog)):
        if not np.isnan(niLog[j]):
            logFramesNi[j, 0] = int(niLog[j])  # niLog[j].split(",")[0]
            logFramesNi[j, 1] = j

    # find the indeces where this movie gave a frame based on the search term
    Inds = []
    for i in range(len(cleanNames)):
        Inds.append(log_df[cleanNames[i]].index[log_df[cleanNames[i]].notna()])
        
    
    #Inds = pd.array(Inds)

    # make smaller database without the other non Ni events
    colNiTimes = {}
    for i in range(len(Inds) - 1):
        # removeInds = pd.Index([])
        # mini_inds = np.setdiff1d(range(len(Inds) - 1), i)
        # dropInds = pd.Index([], dtype=np.int64).append(
        #     Inds[int(mini_inds)]
        # )
        # mini_df = log_df.drop(dropInds).reset_index()

        # get ni frames
        occurenceInds = log_df[cleanNames[i]].index[
            log_df[cleanNames[i]].notna()
        ]

        logTimeValues = log_df.iloc[occurenceInds - 1][cleanNames[-1]].values          
        
        
        # go through cases where there was a video before or after that prevented
        # registering the nidaq time. to take the surest time
        for plus in [-1,-2,-3,-4,1,2,3,4]:
            nanInds = np.where(logTimeValues.astype(str) == "nan")[0]  
            possibleOccurunce = occurenceInds.copy()
            possibleOccurunce = possibleOccurunce[(possibleOccurunce+plus)<(len(log_df)-1)]
            logTimeValues_wherenan = log_df.iloc[possibleOccurunce +plus][cleanNames[-1]].values  
            lostFrames = len(logTimeValues) - len(logTimeValues_wherenan)
            if (lostFrames>0):
                logTimeValues_wherenan = np.append(logTimeValues_wherenan,np.ones(lostFrames)*np.nan)
            logTimeValues[nanInds] = logTimeValues_wherenan[nanInds]
        
            
            
        
        
        
        # for nind in nanInds:
        #     plus = 2            
        #     logTimeValues[nind] = mini_df.iloc[occurenceInds[nind] + plus][
        #         cleanNames[-1]
        #     ]
        #     # carry on until finding the first that is not nan
        #     while str(logTimeValues[nind]) == "nan":
        #         plus += 1
        #         logTimeValues[nind] = mini_df.iloc[occurenceInds[nind] + plus][
        #             cleanNames[-1]
        #         ]
        logFrames = logTimeValues
        # logFrames = np.zeros(len(logTimeValues)) * np.nan
        # for j in range(len(logTimeValues)):
        #     if type(logTimeValues[j]) == str:
        #         # logFrames[j] = logTimeValues[j].split(",")[0]
        #         logFrames[j] =

        colNiTimes[cleanNames[i]] = logFrames
    colNiTimes[cleanNames[-1]] = logFramesNi[:, 0]
    return colNiTimes
