import bisect

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
        time_upward = np.ones((0,1)) * np.nan

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


def detect_wheel_move(move_a, move_b, timestamps, rev_res=1024, total_track=59.847):
    """
    Converts rotary encoder data to velocity and distance travelled.

    Parameters
    ----------
    move_a : np.ndarray
        First channel of the rotary encoder.
    move_b : np.ndarray
        Second channel of the rotary encoder.
    timestamps : np.ndarray
        Timestamps associated with the frames.
    rev_res : int, optional
        Rotary encoder resolution (default: 1024).
    total_track : float, optional
        Total length of the track in cm (default: 59.847).

    Returns
    -------
    velocity : np.ndarray
        Velocity in cm/s.
    """
    # Normalize to binary signals
    move_a = (move_a / np.max(move_a) > 0.5).astype(int)
    move_b = (move_b / np.max(move_b) > 0.5).astype(int)

    # # Detect rising edges in moveA
    # rising_edges_a = np.where(np.diff(move_a, prepend=0) > 0)[0]
    #
    # # Determine direction: +1 if moveB is low, -1 if moveB is high
    # counter = np.zeros(len(move_a))
    # counter[rising_edges_a] = np.where(move_b[rising_edges_a] == 0, 1, -1)

    # Count rising edges
    counter = np.cumsum(np.diff(move_a, prepend=0) > 0)

    # Convert counts to distance
    dist_per_move = total_track / rev_res
    distance = counter * dist_per_move

    # Smooth velocity using Gaussian filter
    dtime = np.nanmedian(np.diff(timestamps, axis=0))
    sigma = int(0.5 / dtime)  # smooth across 0.5 s
    velocity = sp.ndimage.gaussian_filter1d(np.diff(distance, prepend=distance[0]) / dtime, sigma=sigma)

    return velocity


def get_log_entry(log_folder, event_names, time_channel=None):
    """
    Parameters
    ----------
    log_folder : str
        the path of the log file.
    event_names : the string of the entry to look for

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the
        props and their values.

    """
    file = glob.glob(os.path.join(log_folder, "Log*.csv"))
    if len(file) == 0:
        print(f"    No log file found in {log_folder}")
        return None

    event_names = [event_names] if isinstance(event_names, str) else event_names
    data = {name: [] for name in event_names}
    with open(file[0], newline="") as csvfile:
        for i, row in enumerate(csv.reader(csvfile, delimiter=" ", quotechar="|")):
            full_row = "".join(row)
            for pattern in event_names:
                if re.findall(pattern, full_row):
                    data[pattern].append({"row": i, "value": full_row})

    # Convert to dictionary of DataFrames
    data = {name: pd.DataFrame(values) for name, values in data.items()}

    # if time_channel not None, extract all lines numbers where time_channel appears in rows of file
    nt = None
    if time_channel is not None:
        time_rows = []
        with open(file[0], newline="") as csvfile:
            for i, row in enumerate(csv.reader(csvfile, delimiter=" ", quotechar="|")):
                full_row = "".join(row)
                if re.findall(time_channel, full_row):
                    time_rows.append(i)
        nt = len(time_rows)

        for event in event_names:
            # Rename column "row" to "timestamp"
            data[event].rename(columns={"row": "timestamp"}, inplace=True)
            # Update timestamp values based on time_rows
            data[event]["timestamp"] = data[event]["timestamp"].apply(
                lambda row_val: max(0, bisect.bisect_right(time_rows, row_val) - 1)
            )

    return data, nt


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

    # Least-squares: a = m*b + n
    A = np.column_stack([b_matched, np.ones(i1 - i0)])
    result = np.linalg.lstsq(A, a_matched, rcond=None)
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
    if mse > 0.0001:
        print(f"    WARNING: Large mismatch between NiDAQ and Arduino sync signals (MSE={mse:.4f}).")

    time_ard_sync = factor * ardTimes + lag

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


def process_stimulus(titles, directory):
    results, num_trials, protocol, time = stimulus_processing_dictionary[titles](directory)
    return results, num_trials, protocol, time


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


def get_recorded_video_times(log_folder, event_names, time_channel):
    """
    Gets the recorded video times from the log files that bonsai is saving

    Parameters
    ----------
    log_folder : str
        The directory where the log file resides.
    event_names : list (str)
        The terms used to represent the different video recordings .Last has to be the Nidaq.
    output_names : str
        A nicer names to use when logging the entries.

    Returns
    -------
    None.

    """
    event_logs, nt = get_log_entry(log_folder, event_names, time_channel)

    event_times = {event: event_logs[event].timestamp.to_numpy().reshape(-1, 1).astype(float)
                   for event in event_logs.keys()}

    return event_times, nt
