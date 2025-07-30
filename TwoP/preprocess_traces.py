"""Pre-process calcium traces extracted from tiff files."""
import numpy as np
import scipy as sp
from TwoP.general import linear_analytical_solution
import pandas as pd


def correct_neuropil(
    F: np.ndarray,
    N: np.ndarray,
    fs,
    numN=20,
    minNp=10,
    maxNp=90,
    prctl_F=5,
    prctl_F0=5,
    Npil_window_F0=60,
    verbose=True,
):
    """
    Estimates the correction factor r for neuropil correction, so that:
        C = S - rN
        with C: actual signal from the ROI, S: measured signal, N: neuropil

    Parameters
    ----------
    F : np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.
    N : np.ndarray [t x nROIs]
        Neuropil traces of ROIs.
    numN : int, optional
        Number of bins used to partition the distribution of neuropil values.
        Each bin will be associated with a mean neuropil value and a mean
        signal value. The default is 20.
    minNp : int, optional
        Minimum values of neuropil considered, expressed in percentile.
        0 < minNp < 100. The default is 10.
    maxNp : int, optional
        Maximum values of neuropil considered, expressed in percentile.
        0 < maxNp < 100, minNp < maxNp. The
        default is 90.
    prctl_F : int, optional
        Percentile of the measured signal that will be matched to neuropil.
        The default is 5.
    prctl_F0 : int, optional
        Percentile of the measured signal that will be taken as F0.
        The default is 8
    window_F0 : int, optional
        The window size for the calculation of F0 for both signal and neuropil.
        The default is 60.
    verbose : boolean, optional
        Feedback on fitting. The default is True.

    Returns
    -------
    signal : np.ndarray [t x nROIs]
        Neuropil corrected calcium traces.
    regPars : np.ndarray [2 x nROIs], each row: [intercept, slope]
        Intercept and slope of linear fits of neuropil (N) to measured calcium
        traces (F)
    F_binValues : np.array [numN, nROIs]
        Low percentile (prctl_F) values for each calcium trace bin. These
        values were used for linear regression.
    N_binValues : np.array [numN, nROIs]
        Values for each neuropil bin. These values were used for linear
        regression.

    Based on Matlab function estimateNeuropil (in +preproc) written by Mario
    Dipoppa and Sylvia Schroeder
    """

    [nt, nROIs] = F.shape
    N_binValues = np.ones((numN, nROIs)) * np.nan
    F_binValues = np.ones((numN, nROIs)) * np.nan
    regression_pars = np.ones((2, nROIs)) * np.nan
    signal = np.ones((nt, nROIs)) * np.nan

    # Correct for slow drift in ROI and neuropil traces separately.
    F0 = get_F0(F, fs, Npil_window_F0)
    N0 = get_F0(N, fs, Npil_window_F0)
    Fc = F - F0
    Nc = N - N0

    for iROI in range(nROIs):
        iN = Nc[:, iROI]
        iF = Fc[:, iROI]

        # Get range of neuropil values (default: between 10th and 90th percentile) + divide range into numN
        # (default: 20) binsn of equal width.
        N_prct = np.nanpercentile(iN, np.array([minNp, maxNp]), axis=0)
        binSize = (N_prct[1] - N_prct[0]) / numN
        N_binValues[:, iROI] = N_prct[0] + (np.arange(0, stop=numN)) * binSize
        # Associate each neuropil value with a bin number.
        N_ind = np.floor((iN - N_prct[0]) / binSize)

        # Bin ROI signal values the same way as neuropil values. For each bin, find the prctl_F value (default: 5). The
        # idea is to match ROI signal values without spiking activity to the corresponding neuropil values.
        for Ni in range(numN):
            tmp = np.ones_like(iF) * np.nan
            tmp[N_ind == Ni] = iF[N_ind == Ni]
            F_binValues[Ni, iROI] = np.nanpercentile(tmp, prctl_F, 0)
        # Determine relation between neuropil and ROI signal using linear regression.
        noNan = np.where(~np.isnan(F_binValues[:, iROI]) & ~np.isnan(N_binValues[:, iROI]))[0]
        a, b, mse = linear_analytical_solution(
            N_binValues[noNan, iROI], F_binValues[noNan, iROI], False
        )
        # Restrict slope (b) to be between 0 and 2 to avoid over-correction.
        b = min(b, 2)
        b = max(b, 0)
        regression_pars[:, iROI] = (a, b)

        # Correct ROI signal by subtracting prediction based on neuropil.
        signal[:, iROI] = iF - (b * iN + a) + F0[:, iROI]
    return signal, regression_pars, F_binValues, N_binValues


def correct_zmotion(F, N, F_profiles, N_profiles, ztrace, reference_depth, ignore_faults=True, frames_per_experiment=None):
    """
    Corrects changes in fluorescence due to brain movement along z-axis
    (depth). Method is based on algorithm described in Ryan, ..., Lagnado
    (J Physiol, 2020).

    Parameters
    ----------
    F : np.array [t x nROIs]
        Calcium traces (measured signal) of ROIs from a single(!) plane.
        It is assumed that these are neuropil corrected!
    F_profiles : np.array [slices x nROIs]
        Fluorescence profiles of ROIs across depth of z-stack.
        These profiles are assumed to be neuropil corrected!
    ztrace : np.array [t]
        Depth of each frame of the imaged plane.
        Indices in this array refer to slices in zprofiles.
    ignore_faults: bool, optional
        Whether to remove the timepoints where imaging took place in a plane
        that is meaningless to a cell's activity. Default is True.

    threshold: float, optional
        The cutoff for amplification that is deemed too high (thus amplifying noise).
        Default is 0.2 (amplification X5).

    Returns
    -------
    signal : np.array [t x nROIs]
        Z-corrected calcium traces.
    """
    # Determine correction factor for F and N for each slice in Z stack.
    F_factors = F_profiles / F_profiles[reference_depth, :]
    N_factors = N_profiles / N_profiles[reference_depth, :]

    # Disregard corrections for ROI traces when depth is outside the ROI's profile.
    if ignore_faults:
        F_factors = remove_zcorrected_faults(F_factors, reference_depth)
    # TODO: not sure how to deal with the threshold constraint. Threshold relative to what? What should be zero here?
    #  Minimum value of profile is highly dependent on whether zstack contains planes outside of the brain or not.
    #  A limit based on distance from peak in profile makes more sense.
    # # Disregard corrections smaller than threshold.
    # F_factors[(F_factors - F_mins) < threshold] = np.nan
    # Apply correction to F and N.
    correction_F = F_factors[ztrace, :]
    correction_N = N_factors[ztrace, :]
    if frames_per_experiment is None or len(frames_per_experiment) < 2:
        # If there is only one experiment, smooth correction trace for all ROIs.
        correction_F = sp.ndimage.gaussian_filter1d(correction_F, sigma=2, axis=0)
        correction_N = sp.ndimage.gaussian_filter1d(correction_N, sigma=2, axis=0)
    else:
        # If there are multiple experiments, smooth correction trace for each experiment separately.
        lastFrame = 0
        for lf in frames_per_experiment:
            correction_F[lastFrame: lastFrame + lf, :] = sp.ndimage.gaussian_filter1d(
                correction_F[lastFrame: lastFrame + lf, :], sigma=2, axis=0)
            correction_N[lastFrame: lastFrame + lf, :] = sp.ndimage.gaussian_filter1d(
                correction_N[lastFrame: lastFrame + lf, :], sigma=2, axis=0)
            lastFrame += lf
    F_corrected = F / correction_F
    N_corrected = N / correction_N
    return F_corrected, N_corrected


def get_F0(
    Fc, fs, prctl_F=8, window_size=60, framesPerFolder=[], verbose=True
):
    """
    Determines the baseline fluorescence to use for computing deltaF/F.


    Parameters
    ----------
    Fc : np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.
    fs : float
        The frame rate (frames/second/plane).
    prctl_F : int, optional
        The percentile from which to take F0. The default is 8.
    window_size : int, optional
        The rolling window over which to calculate F0. The default is 60.
    framesPerFolder : [frames], optional
        an array with the number of frames in each experiment. if not empty
        then  gets individual F0 for each experiment. default is empty.
    verbose : bool, optional
        Whether or not to provide detailed processing information.
        The default is True.

    Returns
    -------
    F0 : np.ndarray [t x nROIs]
        The baseline fluorescence (F0) traces for each ROI.

    """
    F0 = np.zeros_like(Fc)

    # Determine F0 per experiment (overall fluorescence may change abruptly between experiments).
    if len(framesPerFolder) > 0:
        lastFrame = 0
        for lf in framesPerFolder:
            F0t = get_F0(
                Fc[lastFrame: lastFrame + lf, :],
                fs,
                prctl_F,
                window_size,
                [],
                verbose,
            )
            F0[lastFrame: lastFrame + lf, :] = F0t

    # Determine prctl_F percentile of fluorescence traces in rolling window.
    window_size = int(round(fs * window_size))
    Fc_pd = pd.DataFrame(Fc)
    F0 = np.array(
        Fc_pd.rolling(window_size, min_periods=1, center=True).quantile(
            prctl_F * 0.01
        )
    )
    return F0


def get_delta_F_over_F(Fc, F0):
    """
    Calculates delta F over F. Note instead of simply dividing (F-F0) by F0,
    the mean of F0 is used and only values above 1 are taken. This is to not
    wrongly increase the value of F if F0 is smaller than 1.

    Parameters
    ----------
    Fc :np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.
    F0 : np.ndarray [t x nROIs]
        The baseline fluorescence (F0) traces of ROIs.

    Returns
    -------
    np.ndarray [t x nROIs]
    Change in fluorescence (dF/F) of ROIs.

    """
    # return (Fc - F0) / np.fmax(1, np.nanmean(F0, 0))
    return (Fc - F0) / np.fmax(1, F0)


def remove_zcorrected_faults(zprofiles, reference_depth):
    """
    This function cleans timepoints in the trace where the imaging takes place
    in a plane that is meaningless as to cell activity.
    This is defined as times when there are two peaks or slopes in the imaging
    region and the imaging plane is in the second slope.

    Parameters
    ----------
    ztrace : np.array[t]
        The imaging plane on the z-axis for each frame.
    zprofiles : np.ndarray [z x nROIs]
        Depth profiles of all ROIs.
    signals : np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.

    Returns
    -------
    np.ndarray [t x nROIs]
    signals: the corrected signals with the faulty timepoints removed.

    """
    zprofiles_corrected = np.copy(zprofiles)
    # TODO: for troughs at reference_depth, check to which side from reference the fluorescence increases more steeply.
    #  Then choose thise side.
    for roi in np.arange(zprofiles.shape[1]):
        # Determine indices of all troughs in the z profile.
        trough_inds = sp.signal.argrelmin(zprofiles[:,roi])[0]
        if len(trough_inds) == 0:
            # No troughs found, return the original profile.
            continue
        # Find the two troughs closest to the reference depth.
        neighbor = np.searchsorted(trough_inds, reference_depth)
        # Set Z profile beyond the troughs to NaN.
        if neighbor < len(trough_inds) and trough_inds[neighbor] < zprofiles.shape[0] - 1:
            zprofiles_corrected[trough_inds[neighbor] + 1 :, roi] = np.nan
        if neighbor > 0 and trough_inds[neighbor] > 0:
            zprofiles_corrected[: trough_inds[neighbor-1], roi] = np.nan

    return zprofiles_corrected

# TODO (SS): make zeroValue a user input parameter
def zero_signal(F, zeroValue=19520):
    """

    This function adds the value 19520 to all ROIs across time.This value
    represents the absolute zero signal and was obtained by averaging the
    darkest frame over many imaging sessions. It is important to note
    that the absolute zero value is arbitrary and depends on the voltage range
    of the PMTs.



    Parameters
    ----------
    F : np.ndarray [t x nROIs]
    Calcium traces (measured signal) of ROIs.

    Returns
    -------
    F : np.ndarray [t x nROIs]
    Calcium traces (measured signal) of ROIs with the addition of the absolute
    zero signal.

    """

    return F + zeroValue
