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
from scipy.signal import correlate, correlation_lags

from TwoP import general
from Bonsai_TwoP.behaviour_protocol_functions import *


def get_nidaq_channels(niDaqFilePath, sampling_rate, numChannels=None, plot=False):
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
        channels = np.array([s.lower() for s in channels])
        if len(channels.shape) > 0:
            numChannels = len(channels)
        else:
            numChannels = 1
    else:
        channels = range(numChannels)

    # Gets the actual nidaq file and extract the data from it.
    niDaqFilePath = general.get_file_in_directory(niDaqFilePath, "NidaqInput")
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

    nidaqTime = (np.arange(niDaq.shape[0]) / sampling_rate).reshape(-1, 1)

    return niDaq, channels, nidaqTime


def assign_frame_time(signal: np.ndarray, time: np.ndarray, th=0.5, plot=False):
    """
    Assigns a time in s to a frame time.

    Parameters
    ----------
    signal : np.array[frames]
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
    signal_min = np.nanmin(signal)
    signal_max = np.nanmax(signal)
    threshold = signal_min + (signal_max - signal_min) * th
    # Gets the timepoints where the frame clock crosses a certain threshold.
    idx_upward = np.diff((signal > threshold).astype(int), prepend=0, axis=0) > 0
    time_upward = time[idx_upward]

    # Check whether signal is bimodal as expected from a TTL pulse
    low = signal_min + (signal_max - signal_min) * 0.1
    high = signal_min + (signal_max - signal_min) * 0.9
    mid_band = (signal < low) | (signal > high)
    frac_mid = np.mean(mid_band)
    if frac_mid < 0.9:  # signal not bimodal -> TTL pulse not correctly recorded
        time_upward = None

    if plot:
        f, ax = plt.subplots(1)
        ax.plot(time, signal)
        ax.axhline(threshold, color="k", linestyle="-")
        ax.plot(time_upward, np.ones(len(time_upward)) * threshold, "r^")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Amplitude (V)")

    return time_upward


def detect_photodiode_changes(photodiode, time, plot_folder: str="", kernel=10, upper_threshold=0.7,
                              lower_threshold=0.3, wait_time=10):
    """
    Detects time points when the photodiode signal crosses thresholds.

    Parameters
    ----------
    photodiode : np.array[frames]
        The signal of the photodiode from the nidaq.
    time : np.array[frames]
        Time vector in seconds, same length as photodiode.
    kernel : int
        The kernel for median filtering. The default is 10.
    upper_threshold : float
        Fractional upper threshold (0-1). Downward crossings through this are detected. The default is 0.7.
    lower_threshold : float
        Fractional lower threshold (0-1). Upward crossings through this are detected. The default is 0.3.
    plot_folder: bool
        Plot to inspect. The default is False.
    wait_time: float
        The wait time in seconds. The default is 10.

    Returns
    -------
    crossings_up : np.array
        Upward crossing times (s).
    crossings_down : np.array
        Downward crossing times (s).
    """

    # Median-filter the photodiode signal.
    signal_filtered = sp.signal.medfilt(photodiode[:, 0], kernel_size=int(kernel // 2) * 2 + 1)

    # Scale thresholds to signal range.
    signal_max = np.max(signal_filtered)
    signal_min = np.min(signal_filtered)
    threshold_upper = signal_min + (signal_max - signal_min) * upper_threshold
    threshold_lower = signal_min + (signal_max - signal_min) * lower_threshold

    # Convert wait_time from seconds to sample index.
    wait_time_idx = np.searchsorted(time.flat, wait_time)

    # Detect upward crossings through lower_threshold (signal goes low -> high).
    crossings_up_idx = np.where(np.diff((signal_filtered > threshold_lower).astype(int), prepend=0) > 0)[0]
    crossings_up_idx = crossings_up_idx[crossings_up_idx >= wait_time_idx]

    # Detect downward crossings through upper_threshold (signal goes high -> low).
    crossings_down_idx = np.where(np.diff((signal_filtered > threshold_upper).astype(int), prepend=0) < 0)[0]
    crossings_down_idx = crossings_down_idx[crossings_down_idx >= wait_time_idx]

    # Collect first crossings
    combined = np.hstack([crossings_up_idx, crossings_down_idx])
    crossings_first = np.min(combined) if len(combined) > 0 else time.shape[0]

    # --- Early crossings (after wait_time, before any other crossings, opposite threshold) ---
    # If signal starts below threshold_upper at wait_time: the first event may be a downward
    # crossing of threshold_lower, which belongs to crossings_down.
    early_crossings_down_idx = np.where(np.diff((signal_filtered > threshold_lower).astype(int), prepend=0) < 0)[0]
    early_crossings_down_idx = early_crossings_down_idx[
        (wait_time_idx <= early_crossings_down_idx) & (early_crossings_down_idx < crossings_first)
        ]
    # If signal starts low (below threshold_lower) at wait_time: the first event will be an upward
    # crossing of threshold_upper (not threshold_lower), which belongs to crossings_up.
    early_crossings_up_idx = np.where(np.diff((signal_filtered > threshold_upper).astype(int), prepend=0) > 0)[0]
    early_crossings_up_idx = early_crossings_up_idx[
        (wait_time_idx <= early_crossings_up_idx) & (early_crossings_up_idx < crossings_first)
        ]
    crossings_up_idx = np.hstack([early_crossings_up_idx, crossings_up_idx])
    crossings_down_idx = np.hstack([early_crossings_down_idx, crossings_down_idx])

    # Check that crossings_up_idx and crossings_down_idx alternate.
    # We collect indices from the original arrays that violate alternation.
    all_crossings_idx = np.hstack([crossings_up_idx, crossings_down_idx])
    all_labels = np.hstack([
        np.ones(len(crossings_up_idx), dtype=int),  # 1 -> up
        np.zeros(len(crossings_down_idx), dtype=int),  # 0 -> down
    ])
    all_idx = np.hstack([np.arange(len(crossings_up_idx)), np.arange(len(crossings_down_idx))])

    # Sort by crossing index to get chronological order.
    order = np.argsort(all_crossings_idx)
    sorted_labels = all_labels[order]
    sorted_idx = all_idx[order]

    # Find violations of alternating labels
    label_diffs = np.diff(sorted_labels)
    violations = np.where(label_diffs == 0)[0]

    # Remove violating crossings (1st crossing in same direction); store trial index of violation
    violation_up = []
    violation_down = []
    for i in violations:
        if sorted_labels[i] == 0:  # 2 downward crossings in a row
            violation_down.append(sorted_idx[i])
        else:  # 2 upward crossings in a row
            violation_up.append(sorted_idx[i])

    violation_down = np.array(violation_down, dtype=int)
    deleted_down = crossings_down_idx[violation_down]
    crossings_down_idx = np.delete(crossings_down_idx, violation_down)
    violation_up = np.array(violation_up, dtype=int)
    deleted_up = crossings_up_idx[violation_up]
    crossings_up_idx = np.delete(crossings_up_idx, violation_up)

    # Correct indices for deletions
    violation_up = violation_up - np.arange(len(violation_up))
    violation_down = violation_down - np.arange(len(violation_down))

    # Convert indices back to time values.
    crossings_up = time[crossings_up_idx]
    crossings_down = time[crossings_down_idx]
    violation_up = time[crossings_up_idx[violation_up]]
    violation_down = time[crossings_down_idx[violation_down]]

    if os.path.isdir(plot_folder):
        for i in range(len(violation_up)):
            idx = deleted_up[i] + np.arange(-10000, 10000)
            idx = idx[(idx >= 0) & (idx < len(time))]  # keep valid indices

            plt.figure(figsize=(16,8))
            plt.plot(time[idx], signal_filtered[idx], label="photodiode filtered")
            plt.plot(violation_up[i], threshold_lower, "g^", label="upward crossing")
            plt.plot(time[deleted_up[i]], threshold_lower, "ro", label="deleted crossing")
            plt.axhline(threshold_upper, color="k", linestyle="-")
            plt.axhline(threshold_lower, color="k", linestyle="-")
            plt.legend()
            plt.xlabel("time (s)")
            plt.savefig(os.path.join(plot_folder, f"photodd_violation_up_{i}.png"))
            plt.close()

        for i in range(len(violation_down)):
            idx = deleted_down[i] + np.arange(-10000, 10000)
            idx = idx[(idx >= 0) & (idx < len(time))]  # keep valid indices

            plt.figure(figsize=(16,8))
            plt.plot(time[idx], signal_filtered[idx], label="photodiode filtered")
            plt.plot(violation_down[i], threshold_upper, "gv", label="downward crossing")
            plt.plot(time[deleted_down[i]], threshold_upper, "ro", label="deleted crossing")
            plt.axhline(threshold_upper, color="k", linestyle="-")
            plt.axhline(threshold_lower, color="k", linestyle="-")
            plt.legend()
            plt.xlabel("time (s)")
            plt.savefig(os.path.join(plot_folder, f"photodd_violation_down_{i}.png"))
            plt.close()

    return crossings_up, crossings_down, violation_up, violation_down


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

    # for older recordings
    # check if signals are the same or delayed
    similarity = np.sum(moveA == ~moveB)/len(moveB)

    # Detects A move.
    risingEdgeA = np.where(np.diff(moveA > 0, prepend=True))[0]
    risingEdgeA = risingEdgeA[moveA[risingEdgeA] == 1]
    risingEdgeA_B = moveB[risingEdgeA]
    counterA[risingEdgeA[risingEdgeA_B == 0]] = 1
    counterA[risingEdgeA[risingEdgeA_B == 1]] = -1

    if (not (similarity > 0.9)):
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
    averagingTime = int(np.round(1 / np.nanmedian(np.diff(timestamps))))
    sumKernel = np.ones(averagingTime)
    tsKernel = np.zeros(averagingTime)
    tsKernel[0] = 1
    tsKernel[-1] = -1

    # Taking the difference and
    distDiff = np.diff(distance, prepend=True)
    velocity = (
        sp.ndimage.gaussian_filter1d(distDiff, averagingTime / 2)
        * averagingTime
    )
    velocity[0] = np.nanmedian(velocity)
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


# @jit(forceobj=True)
def get_arduino_data(arduinoDirectory, sampling_rate, plot=False):
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
    arduinoFilePath = general.get_file_in_directory(arduinoDirectory, "ArduinoInput")
    # Loads the arduino data.
    csvChannels = np.loadtxt(arduinoFilePath, delimiter=",")
    # convert time to second (always in ms)
    arduinoTime = (csvChannels[:, -1] / sampling_rate).reshape(-1, 1)

    # Starts arduino time at zero.
    arduinoTime -= arduinoTime[0]
    # Takes all channels except the timepoints.
    csvChannels = csvChannels[:, :-1]
    numChannels = csvChannels.shape[1]  # Gets number of channels.
    if plot:  # Option to plot all channels.
        f, ax = plt.subplots(numChannels, sharex=True)
        for i in range(numChannels):
            ax[i].plot(arduinoTime, csvChannels[:, i])

    # Gets the names of each channel from the channel csv file.
    dirs = glob.glob(os.path.join(arduinoDirectory, "arduinoChannels*.csv"))
    if len(dirs) == 0:
        channelNames = []
    else:
        channelNames = np.loadtxt(dirs[0], delimiter=",", dtype=str)
        channelNames = np.array([s.lower() for s in channelNames])

    return csvChannels, channelNames, arduinoTime


def estimate_sync_lag_xcorr_fast(a, b):

    a = np.asarray(a, float)
    b = np.asarray(b, float)

    da = np.diff(a, axis=0)
    db = np.diff(b, axis=0)

    da = (da - da.mean()) / da.std()
    db = (db - db.mean()) / db.std()

    corr = correlate(db, da, mode="full", method="fft")
    lags = correlation_lags(len(db), len(da), mode="full")

    pulse_shift = int(lags[np.argmax(corr)])
    i0 = max(0, -pulse_shift)
    i1 = min(len(a), len(b) - pulse_shift)

    a_matched = a[i0:i1]
    b_matched = b[i0 + pulse_shift : i1 + pulse_shift]

    # Least-squares: b = m*a + n
    A = np.column_stack([a_matched, np.ones(i1 - i0)])
    result = np.linalg.lstsq(A, b_matched, rcond=None)
    factor, lag = result[0]

    residuals = result[1]
    mse = float(residuals[0]) / (i1 - i0) if len(residuals) > 0 else np.nan

    return factor, lag, mse


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

    # Gets where the ni sync signal changes.
    niChange = np.where(np.diff(niTick, prepend=False) > 0)[0]
    niChangeTime = niTimes[niChange]

    # Gets where the arduino sync signal changes.
    ardChange = np.where(np.diff(ardTick, prepend=False) > 0)[0]
    ardChangeTime = ardTimes[ardChange]

    factor, lag, mse = estimate_sync_lag_xcorr_fast(niChangeTime, ardChangeTime)
    if mse > 0.01:
        print(f"    WARNING: Large mismatch between NiDAQ and Arduino sync signals (MSE={mse:.4f}).")

    time_ard_sync = factor * ardChangeTime + lag

    #     lastPoint = 0
    #     # Within this for loop, finds where there are misalignments due to
    #     # potentially uneven acquisition of the signal and realigns it.
    #     for i in range(0, len(ardChangeTime) + 1, batchSize):
    #         if i >= len(ardChangeTime):
    #             continue
    #
    #         x = ardChangeTime[i: np.min([len(ardChangeTime), i + batchSize])]
    #         y = niChangeTime[i: np.min([len(ardChangeTime), i + batchSize])]
    #
    #         a, b, mse = linear_analytical_solution(x, y)
    #
    #         ind = np.where((newArdTimes >= lastPoint))[0]
    #         newArdTimes[ind] = b * newArdTimes[ind] + a
    #
    #         ardChangeTime = ardChangeTime * b + a
    #
    #         lastPoint = (
    #             ardChangeTime[np.min([len(ardChangeTime) - 1, i + batchSize])]
    #             + 0.00001
    #         )
    return time_ard_sync


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
    piezo : np.array [miliseconds in one frame, nplanes]
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
        piezoStarts = frameTimes[imagingPlanes + plane:: imagingPlanes]
        piezoEnds = frameTimes[imagingPlanes + plane + 1:: imagingPlanes]

        # Determines the range over which to sample over the piezo trace given the batchFactor specified.
        piezoBatchRange = range(
            0, min(len(piezoStarts), len(piezoEnds)), batchFactor)
        # Creates the array for the piezo location for each milisecond in each batch.
        avgTrace = np.zeros((traceDuration, len(piezoBatchRange)))
        for avgInd, pi in enumerate(piezoBatchRange):
            # Determines the section of the piezo trace to take into account given the piezo start and end times
            # specified above.
            # TODO (SS): Check that this still works if length(inds) is not equal to len(avgTrace[:, avgInd]).
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
    piezo : np.array [miliseconds in one frame, nplanes]
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


def process_stimulus(titles, directory):
    results, num_trials, protocol = stimulus_processing_dictionary[titles](directory)
    return results, num_trials, protocol


def save_stimuli(saveDirectory, stimulusTypes, stimulusProps):
    stimulusTypes = np.array(stimulusTypes)
    stimulusProps = np.array(stimulusProps)
    uniqueType = np.unique(stimulusTypes)

    for t in stimulusTypes:
        props = stimulusProps[stimulusTypes == t]
        props_df = pd.DataFrame.from_records(props)
        fileNames = props_df.columns.values.astype(str)
        # save all filenames in the save directory
        for f in fileNames:
            np.save(os.path.join(saveDirectory, f), np.vstack(props_df[f]))


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

    # check if name exists in log if not remove
    removedNames = []
    removedInds = []
    for i in range(len(cleanNames)):
        if (not cleanNames[i] in log_df.keys()):
            removedNames.append(cleanNames[i])
            removedInds.append(i)
    for i in removedInds:
        cleanNames.pop(i)

    # find the indeces where this movie gave a frame based on the search term
    Inds = []
    for i in range(len(cleanNames)):
        if (cleanNames[i] in log_df.keys()):
            Inds.append(
                log_df[cleanNames[i]].index[log_df[cleanNames[i]].notna()])
        else:
            Inds.append(
                log_df[cleanNames[1]].index[log_df[cleanNames[i]].notna()])

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
        for plus in [-1, -2, -3, -4, 1, 2, 3, 4]:
            nanInds = np.where(logTimeValues.astype(str) == "nan")[0]
            possibleOccurunce = occurenceInds.copy()
            possibleOccurunce = possibleOccurunce[(
                possibleOccurunce+plus) < (len(log_df)-1)]
            logTimeValues_wherenan = log_df.iloc[possibleOccurunce +
                                                 plus][cleanNames[-1]].values
            lostFrames = len(logTimeValues) - len(logTimeValues_wherenan)
            if (lostFrames > 0):
                logTimeValues_wherenan = np.append(
                    logTimeValues_wherenan, np.ones(lostFrames)*np.nan)
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
    for r in removedNames:
        colNiTimes[r] = np.ones(len(logTimeValues))*np.nan
    return colNiTimes
