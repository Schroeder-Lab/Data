# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:39:57 2022

@author: LABadmin
"""

"""Runner functions"""

from suite2p.registration.zalign import compute_zpos
from joblib import Parallel, delayed
import numpy as np
import time
import traceback
import io
import os
import cv2
import skimage.io
import glob
import pickle
import scipy as sp
import warnings

from Data.TwoP.process_tiff import *
from Data.TwoP.preprocess_traces import *
from Data.Bonsai.extract_data import *
from Data.TwoP.general import *
from Data.user_defs import create_2p_processing_ops, directories_to_register


def _process_s2p_singlePlane(
    pops, planeDirs, zstackPath, saveDirectory, piezo, plane
):
    """


    Parameters
    ----------
    pops : dict [6]
        The dictionary with all the processing infomration needed. Refer to the
        function create_processing_ops in user_defs for a more in depth
        description.
    planeDirs : list [str of directories]
        List containing the directories refering to the plane subfolders in the
        suite2p folder.
    zstackPath : str [zStackDir\Animal\Z stack folder\Z stack.tif]
        The path of the acquired z-stack.
    saveDirectory : str, optional
        the directory where the processed data will be saved.
        If None will add a ProcessedData directory to the suite2pdir.
        The default is None.
    piezo : np.array [miliseconds in one frame, nplanes]
        Movement of piezo across z-axis for all planes.
        Location in depth (in microns) is for each milisecond within one plane.
    plane : int
        The current plane to process.

    Returns
    -------
    results : dict [5]
        Returns a dictinary which contains the
        - deltaF/F traces: np.array[total frames, ROIs]
        - dF/F Z corrected traces: np.array[total frames, ROIs]
        - z profiles of each ROI: np.array[z x nROIs]
        - Z trace (which indicate the location of the imaging plane over time):
            np.array[frames]
        - Cell locations in Y, X and Z: np.array[no. of ROIs, 3]

    """
    # Sets the current plane to processed.
    if plane > len(planeDirs) - 1:
        return None
    currDir = planeDirs[plane]
    # Array of fluorescence traces [ROIs x timepoints].
    F = np.load(os.path.join(currDir, "F.npy"), allow_pickle=True).T
    # Array of neuropil traces [ROIs x timepoints].
    N = np.load(os.path.join(currDir, "Fneu.npy")).T
    # Array to determine if an ROI is a cell [ROIs].
    isCell = np.load(os.path.join(currDir, "iscell.npy")).T
    # Array of objects with statistics computed for each cell [ROIs]
    stat = np.load(os.path.join(currDir, "stat.npy"), allow_pickle=True)
    #  Dictionary of options and intermediate outputs.
    ops = np.load(os.path.join(currDir, "ops.npy"), allow_pickle=True).item()
    # TODO: why an empty dict here?
    processing_metadata = {}
    
    # remove bad frames
    badFrames = ops['badframes'] 
    F[badFrames,:] = np.nan
    N[badFrames,:] = np.nan
    
    # Gets the acquisition frame rate.
    fs = ops["fs"]
    # Updates F to only include the ROIs considered cells.
    F = F[:, isCell[0, :].astype(bool)]
    # Updates N to only include the ROIs considered cells.
    N = N[:, isCell[0, :].astype(bool)]
    # Updates stat to only include the ROIs considered cells.
    stat = stat[isCell[0, :].astype(bool)]

    # Creates array to place the X, Y and Z positions of ROIs.
    cellLocs = np.zeros((len(stat), 3))
    # Gets the resolution (in pixels) along the y dimension.
    if type(ops["refImg"]) is np.ndarray:
        ySpan = ops["refImg"].shape[1]
    if type(ops["refImg"]) is list:
        ySpan = ops["refImg"][0].shape[1]
    if pops["absZero"] is None:
        # Takes default darkest value
        # Adds the absolute signal value to F, see function for a more details.
        F = zero_signal(F)
        # Adds the absolute signal value to N, see function for a more details.
        N = zero_signal(N)
    else:
        # Takes user darkest value
        # Adds the absolute signal value to F, see function for a more details.
        F = zero_signal(F, pops["absZero"])
        # Adds the absolute signal value to N, see function for a more details.
        N = zero_signal(N, pops["absZero"])

    # For each ROI, the location is determined from the suite2p output "stat"
    # (for X and Y) and from the piezo (for Z).
    for i, s in enumerate(stat):
        # Determines the relative Y position in the FOV by getting the
        # location in pixels of the center of the ROI and divides this by the
        # total resolution.
        relYpos = s["med"][1] / ySpan
        # Due to the fast volume scanning technique used (with a piezo),
        # the plane is imaged at a slant which spans the Y dimension.
        # So the location of the cell in Z depends on its position in Y.
        # For each plane, the piezo array contains the location in Z as it
        # scans through the plane. To determine the correct Z location,
        # the relative Y position was computed in the previous line to compute
        # the index in the piezo array which corresponds to the ROIs location.
        piezoInd = int(np.round((len(piezo) - 1) * relYpos))
        # Determines the Z position of the ROI based on the index calculated
        # in the previous line.
        zPos = piezo[piezoInd]
        # Appends the array with the YX positions of the center of the ROI
        # taken from the stat array and the z position of each ROI.
        # NOTE: Suite2P outputs the positions in XY as [Y,X], need to be kept in
        # mind when wanting to associate a cell with it's location in the FOV
        # as the assumed order would usually be [X,Y].
        cellLocs[i, :] = np.append(s["med"], zPos)

    # Calculates the corrected neuropil traces and the specific values that
    # were used to determine the correction factor (intercept and slope of
    # linear fits, F traces bin values, N traces bin values). Refer to function
    # for further details.
    Fc, regPars, F_binValues, N_binValues = correct_neuropil(
        F,
        N,
        fs,
        prctl_F0=pops["f0_percentile"],
        Npil_window_F0=pops["Npil_f0_window"],
    )
    # Calculates the baseline fluorescence F0 used to calculate delta F over F.
    F0 = get_F0(
        Fc,
        fs,
        prctl_F=pops["f0_percentile"],
        window_size=pops["f0_window"],
        framesPerFolder=ops["frames_per_folder"],
    )
    # Calculates delta F oer F given the corrected neuropil traces and the
    # baseline fluorescence.
    dF = get_delta_F_over_F(Fc, F0)

    # Multi-step process for Z correction.
    zprofiles = None  # Creates NoneType object to place the z profiles.
    zTrace = None  # Creates NoneType object to place the z traces.
    # Specifies the current directory as the path to the registered binary and
    # ops file (Hack to avoid random reg directories).
    ops["reg_file"] = os.path.join(currDir, "data.bin")
    ops["ops_path"] = os.path.join(currDir, "ops.npy")
    
    isZcorrected = np.zeros(F.shape[1]).astype(bool)
    
    # Unless there is no Z stack path specified, does Z correction.
    if not (zstackPath is None):
        try:
            refImg = ops["meanImg"]  # Gets the reference image from Suite2P.
            # Creates registered Z stack path.
            zFileName = os.path.join(
                saveDirectory, "zstackAngle_plane" + str(plane) + ".tif"
            )
            # Registers Z stack unless it was already registered and saved.
            if not (os.path.exists(zFileName)):
                zstack = register_zstack(
                    zstackPath,
                    spacing=1,
                    piezo=piezo,
                    target_image=refImg,
                    channel=1#ops["align_by_chan"],
                )
                # Saves registered Z stack in the specified or default saveDir.
                skimage.io.imsave(zFileName, zstack)
                # Calculates how correlated the frames are with each plane
                # within the Z stack (suite2p function).
                _, zcorr = compute_zpos(zstack, ops)
            # Calculates Z correlation if Z stack was already registered.
            elif not ("zcorr" in ops.keys()):
                zstack = skimage.io.imread(zFileName)
                # Calculates how correlated the frames are with each plane
                # within the Z stack (suite2p function).
                ops, zcorr = compute_zpos(zstack, ops)
                # Saves the current ops path to the ops file.
                np.save(ops["ops_path"], ops)
            # If the Z stack has been registered and Z correlation has been
            # done, loads the saved registered Z stack and the Z correlation
            # values from the ops.
            else:
                zstack = skimage.io.imread(zFileName)
                zcorr = ops["zcorr"]
            # Gets the location of each frame in Z based on the highest
            # correlation value.
            zTrace = np.argmax(zcorr, 0)
            # Computes the Z profiles for each ROI.
            zprofiles = extract_zprofiles(
                currDir,
                zstack,
                neuropil_correction=regPars[:, :],
                metadata=processing_metadata,
                smoothing_factor=2,
                abs_zero = pops["absZero"])
            
            # quantify how many z profiles are at 0 (meaning the neuropil was stronger)
            isZcorrected = ~np.all(zprofiles==0,axis=0)
            # Corrects traces for z motion based on the Z profiles.
            Fcz = correct_zmotion(
                dF,
                zprofiles,
                zTrace.copy(),
                ignore_faults=pops["remove_z_extremes"],
                metadata=pops,
            )
        except:
            # If there is an error in processing, the uncorrected delta F over
            # F is considered.
            print(currDir + ": Error in correcting z-motion")
            print(traceback.format_exc())
            Fcz = dF
    else:
        # If no Z correction is performed (for example if no Z stack was given)
        # only the uncorrected delta F over F is considered.
        Fcz = dF
        zcorr = np.nan
    # Places all the results in a dictionary (dF/F, Z corrected dF/F,
    # z profiles, z traces and the cell locations in X, Y and Z).
    results = {
        "dff": dF,
        "dff_zcorr": Fcz,
        "zProfiles": zprofiles,
        "zTrace": zTrace,
        "zCorr_stack": zcorr,
        "locs": cellLocs,
        "isZcorrected": isZcorrected,
        "cellId": np.where(isCell[0, :].astype(bool))[0],
    }

    if pops["plot"]:
        for i in range(dF.shape[-1]):
            # Print full
            plotArrangement = [
                ["profile", "f"],
                ["profile", "corr"],
                ["profile", "zcorr"],
                ["profile", "trace"],
            ]
            f, ax = plt.subplot_mosaic(plotArrangement)
            ax["f"].plot(F[:, i], "b")
            ax["f"].plot(N[:, i], "r")
            ax["f"].legend(
                ["Fluorescence", "Neuropil"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["corr"].plot(Fc[:, i], "k")
            ax["corr"].plot(F0[:, i], "b", linewidth=4, zorder=10)
            ax["corr"].legend(
                ["Corrected F", "F0"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )

            ax["zcorr"].plot(dF[:, i], "b", linewidth=3)
            ax["zcorr"].plot(Fcz[:, i], "k", alpha=0.5)
            ax["zcorr"].legend(
                ["dF/F", "dF/F z-zcorrected"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["zcorr"].set_xlabel("time (frames)")
            if not zTrace is None:
                ax["trace"].plot(zTrace)
                ax["trace"].legend(
                    ["Z trace"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
            if not zprofiles is None:
                ax["profile"].plot(zprofiles[:, i], range(zprofiles.shape[0]))
                ax["profile"].legend(
                    ["Z profile"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
                ax["profile"].set_xlabel("fluorescence")
                ax["profile"].set_xlabel("depth")

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.savefig(
                os.path.join(
                    saveDirectory,
                    "Plane" + str(plane) + "Neuron" + str(i) + ".png",
                ),
                format="png",
            )

            with open(
                os.path.join(
                    saveDirectory,
                    "Plane" + str(plane) + "Neuron" + str(i) + ".fig.pickle",
                ),
                "wb",
            ) as file:
                pickle.dump(f, file)
            # Print Part
            f, ax = plt.subplot_mosaic(plotArrangement)
            ax["f"].plot(F[1:500, i], "b")
            ax["f"].plot(N[1:500, i], "r")
            ax["f"].legend(
                ["Fluorescence", "Neuropil"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["corr"].plot(Fc[1:500, i], "k")
            ax["corr"].plot(F0[1:500, i], "b", linewidth=4)
            ax["corr"].legend(
                ["Corrected F", "F0"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["zcorr"].plot(dF[1:500, i], "b", linewidth=3)
            ax["zcorr"].plot(Fcz[1:500, i], "k", alpha=0.3)
            ax["zcorr"].legend(
                ["dF/F", "dF/F z-zcorrected"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["zcorr"].set_xlabel("time (frames)")
            if not zTrace is None:
                ax["trace"].plot(zTrace[1:500])
                ax["trace"].legend(
                    ["Z trace"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
            if not zprofiles is None:
                ax["profile"].plot(zprofiles[:, i], range(zprofiles.shape[0]))
                ax["profile"].legend(
                    ["Z profile"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
                ax["profile"].set_xlabel("fluorescence")
                ax["profile"].set_xlabel("depth")

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            plt.savefig(
                os.path.join(
                    saveDirectory,
                    "Plane" + str(plane) + "Neuron" + str(i) + "_zoom.png",
                ),
                format="png",
            )

            with open(
                os.path.join(
                    saveDirectory,
                    "Plane"
                    + str(plane)
                    + "Neuron"
                    + str(i)
                    + "_zoom.fig.pickle",
                ),
                "wb",
            ) as file:
                pickle.dump(f, file)

            plt.close("all")
    return results


def process_s2p_directory(
    suite2pDirectory,
    pops=create_2p_processing_ops(),
    piezoTraces=None,
    zstackPath=None,
    saveDirectory=None,
    ignorePlanes=None,
    debug=False,
):
    """
    This function runs over a suite2p directory and pre-processes the data in
    each plane the pre processing includes:
        neuropil correction
        z-trace extraction and correction according to profile
        at the function saves all the traces together

    Parameters
    ----------
    suite2pDirectory : str [s2pDir/Animal/Date/suite2p]
        the suite2p parent directory, where the plane directories are.
    piezoTraces : [time X plane] um
        a metadata directory for the piezo trace.
    zstackPath : str [zStackDir\Animal\Z stack folder\Z stack.tif]
        the path of the acquired z-stack.
    saveDirectory : str, optional
        the directory where the processed data will be saved. If None will add
        a ProcessedData directory to the suite2pdir. The default is None.

    Returns
    -------
    None.

    """
    if saveDirectory is None:
        # Creates the directory where the processed data will be saved.
        saveDirectory = os.path.join(suite2pDirectory, "ProcessedData")
    if not os.path.isdir(saveDirectory):
        os.makedirs(saveDirectory)
    # Creates a list which contains the directories to the subfolders for each
    # plane.
    planeDirs = glob.glob(os.path.join(suite2pDirectory, "plane*"))   
    # Loads the ops dictionary from the combined directory.
    ops = np.load(
        os.path.join(planeDirs[0], "ops.npy"), allow_pickle=True
    ).item()
    # Loads the number of planes into a variable.
    numPlanes = ops["nplanes"]
    # Creates an array with the plane range.
    planeRange = np.arange(numPlanes)

    # Removes the ignored plane (if specified) from the plane range array.
    if not (ignorePlanes is None):
        ignorePlanes = np.intersect1d(planeRange, ignorePlanes)
        planeRange = np.delete(planeRange, ignorePlanes)
    # Determine the absolute time before processing.
    preTime = time.time()

    # Specifies the amount of parallel jobs to decrease processing time.
    # If in debug mode, there will be no parallel processing.
    if not debug:
        jobnum = 4
    else:
        jobnum = 1
    # Processes the 2P data for the planes specified in the plane range.
    # This gives a list of dictionaries with all the planes.
    # Refer to the function for a more thorough description.
    results = Parallel(n_jobs=jobnum, verbose=5)(
        delayed(_process_s2p_singlePlane)(
            pops, planeDirs, zstackPath, saveDirectory, piezoTraces[:, p], p
        )
        for p in planeRange
    )
    # signalList = _process_s2p_singlePlane(planeDirs,zstackPath,saveDirectory,piezoTraces[:,0],1)
    # Determines the absolute time after processing.
    postTime = time.time()
    print("Processing took: " + str(postTime - preTime) + " ms")

    # Creates lists to place the outputs from the function
    # _process_s2p_singlePlane.
    planes = np.array([])

    signalList = []
    signalLocs = []
    zTraces = []
    zProfiles = []
    isZcorrectedList= []
    zCorrs = []
    cellIds = []
    # Appends lists with the results for all the planes.
    for i in range(len(results)):
        if not (results[i] is None):
            signalList.append(results[i]["dff_zcorr"])
            signalLocs.append(results[i]["locs"])
            zTraces.append(results[i]["zTrace"])
            zProfiles.append(results[i]["zProfiles"])
            zCorrs.append(results[i]["zCorr_stack"])
            cellIds.append(results[i]["cellId"])
            isZcorrectedList.append(results[i]["isZcorrected"])
            # Places the signal into an array.
            res = signalList[i]
            # Specifies which plane each ROI belongs to.
            planes = np.append(planes, np.ones(res.shape[1]) * planeRange[i])
    # Specifies number to compare the length of the signals to.
    minLength = np.inf
    for i in range(len(signalList)):
        # Checks the minumum length of the signals for each plane.
        minLength = np.min((signalList[i].shape[0], minLength))
    for i in range(len(signalList)):
        # Updates the signalList to only include frames until the minimum
        # length determined above.
        # This is done to discard any additional frames that were recorded for
        # some planes but not all.
        signalList[i] = signalList[i][: int(minLength), :]
        if not zTraces[i] is None:
            # Updates the zTraces and zCorrs to only include frames until the minimum
            # length determined above.
            zTraces[i] = zTraces[i][: int(minLength)]
            zCorrs[i] = zCorrs[i][: int(minLength)]
    # Combines results from each plane into a single array for signals,
    # locations, zProfile and zTrace.
    signals = np.hstack(signalList)
    locs = np.vstack(signalLocs)
    zProfile = np.hstack(zProfiles)
    zTrace = np.vstack(zTraces)
    zCorrs = np.swapaxes(np.dstack([a,a,a]).T,1,2)
    cellIds = np.hstack(cellIds)
    isZcorrected = np.hstack(isZcorrectedList)

    # Saves the results as individual npy files.
    np.save(os.path.join(saveDirectory, "calcium.dff.npy"), signals)
    np.save(os.path.join(saveDirectory, "rois.planes.npy"), planes)
    np.save(os.path.join(saveDirectory, "rois.id.npy"), cellIds)
    np.save(os.path.join(saveDirectory, "rois.xyz.npy"), locs)
    np.save(os.path.join(saveDirectory, "rois.zProfiles.npy"), zProfile.T)
    np.save(os.path.join(saveDirectory, "calcium.isZCorrected.npy"), isZcorrected)
    np.save(os.path.join(saveDirectory, "planes.zTrace"), zTrace)
    np.save(os.path.join(saveDirectory, "planes.zcorrelation"), zCorrs)


# bonsai + arduino
# TODO: comment
def process_metadata_directory(
    bonsai_dir, ops=None, pops=create_2p_processing_ops(), saveDirectory=None
):
    """

    Processes all the metadata obtained. Assumes the metadata was recorded with
    two separated devices (in our case a niDaq and an Arduino). The niDaq was
    used to record the photodiode changes,the frameclock, pockels, piezo
    movement, lick detection (if a reward experiment was performed) and a sync
    signal (to be able to synchronise it to the other device). The Arduino was
    used to record the wheel movement, the camera frame times and a syn signal
    to be able to synchronise with the niDaq time.

    The metadata processed and/or reorganised here includes:
    - the times of frames, wheel movement and camera frames
    - sparse noise metadata: start + end times and maps
    - retinal classification metadata: start + end times and stim details
    - circles metadata: start + end times and stim details
    - gratings metadata: start + end times and stim details
    - velocity of the wheel (= mouse running velocity)

    Please have a look at the Bonsai files for the specific details for each
    experiment type.


    Parameters
    ----------
    bonsai_dir : str
        The directory where the metadata is saved.
    ops : dict
        The suite2p ops file.
    pops : dict [6], optional
        The dictionary with all the processing infomration needed. Refer to the
        function create_processing_ops in user_defs for a more in depth
        description.
    saveDirectory : str, optional
        the directory where the processed data will be saved.
        If None will add a ProcessedData directory to the suite2pdir. The
        default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if saveDirectory is None:
        saveDirectory = os.path.join(suite2pDirectory, "ProcessedData")
    # metadataDirectory_dirList = glob.glob(os.path.join(metadataDirectory,'*'))

    metadataDirectory_dirList = ops["data_path"]

    # Gets the length of each experiment in frames.
    fpf = ops["frames_per_folder"]
    # Gets how many planes were imaged.
    planes = ops["nplanes"]
    lastFrame = 0

    # Prepares the lists of outputs.

    # Recordings times, rotary encoder times, camera times.
    frameTimes = []
    wheelTimes = []
    faceTimes = []
    bodyTimes = []

    # The velocity given by the rotary encoder information.
    velocity = []

    # The sparse noise start + end times and the RF maps.
    sparseSt = []
    sparseEt = []
    sparseMaps = []
    sparseEdges = []

    # Retinal protocol stimulus start + end times and stim identity.
    retinalSt = []
    retinalEt = []
    retinalStim = []

    # Gratings start + end times + stim identities for all the different params.
    gratingsSt = []
    gratingsEt = []
    gratingsOri = []
    gratingsSfreq = []
    gratingsTfreq = []
    gratingsContrast = []
    gratingsReward = []

    # Circles start + end times + stim identities for all the different params.
    circleSt = []
    circleEt = []
    circleX = []
    circleY = []
    circleDiameter = []
    circleWhite = []
    circleDuration = []
    
    # natural images file, start and end time
    naturalSt = []
    naturalEt = []
    naturalFiles = []
    

    for dInd, di in enumerate(metadataDirectory_dirList):
        sparseNoise = False
        print(f"Directory: {di}")
        if len(os.listdir(di)) == 0:
            continue
        # Moves on if not a directory (even though ideally all should be a dir).
        # if (not(os.path.isdir(di))):
        #     continue
        expDir = os.path.split(di)[-1]

        # if folder is not selected for analysis move on
        # if not(expDir.isnumeric()) or not (int(expDir) in folder_numbers):
        #     continue

        # frame_in_file = fpf[int(expDir) - 1]
        # In case there are more metadata directories than the experiments that
        # were processed with suite2p, skips these.
        if dInd >= len(fpf):
            warnings.warn(
                "More metadata directories than frames per folder in ops. skipping the rest"
            )
            continue
        # Gets the number of frames in the current experiment to be processed.
        frame_in_file = fpf[dInd]

        try:
            # Gets all the niDaq data, the number of channels and the niDaq
            # frame times.
            nidaq, chans, nt = get_nidaq_channels(di, plot=pops["plot"])
        except Exception as e:
            print("Error is directory: " + di)
            print("Could not load nidaq data")
            print(e)
        try:
            # Gets the frame clock data.
            frameclock = nidaq[:, chans == "frameclock"]
            # Assigns a time in ms to a frame time (see function for details).
            frames = assign_frame_time(frameclock, plot=pops["plot"])
            # TODO: run the 5 lines below in debug mode.
            frameDiffMedian = np.median(np.diff(frames))
            # Take only first frames of each go.
            firstFrames = frames[::planes]
            imagedFrames = np.zeros(frame_in_file) * np.nan
            imagedFrames[: len(firstFrames)] = firstFrames
            planeTimeDelta = np.arange(planes) * frameDiffMedian
        except:
            print("Error is directory: " + di)
            print("Could not extract frames, filling up with NaNs")
            frameTimes.append(np.zeros(frame_in_file) * np.nan)
            continue
        # Adds the frame times to the frameTimes list.
        frameTimes.append(imagedFrames + lastFrame)

        # Gets the sparse noise file snf the props file (with the experimental
        # details) for mapping RFs.
        sparseFile = glob.glob(os.path.join(di, "SparseNoise*"))
        propsFile = glob.glob(os.path.join(di, "props*.csv"))
        propTitles = np.loadtxt(
            propsFile[0], dtype=str, delimiter=",", ndmin=2
        ).T

        if propTitles[0] == "Spont":
            sparseNoise = True

        try:
            # Gets the photodiode data.
            photodiode = nidaq[:, chans == "photodiode"]
            # Gets the frames where photodiode changes are detected.
            frameChanges = detect_photodiode_changes(
                photodiode, plot=pops["plot"]
            )
            frameChanges += lastFrame

            # TODO: Have one long st and et list with different identities so a
            # list of st,et and a list with the event type

            # Gets the sparse map.
            if len(sparseFile) != 0:
                sparseMap = get_sparse_noise(di)
                sparseMap = sparseMap[: len(frameChanges), :, :]
                sparseNoise = True

                # Calculates the end of the final frame.
                sparse_et = np.append(
                    frameChanges[1::],
                    frameChanges[-1] + np.median(np.diff(frameChanges)),
                )
                # Adds the data from above to the respective lists.
                sparseSt.append(frameChanges.reshape(-1, 1).copy())
                sparseEt.append(sparse_et.reshape(-1, 1).copy())
                sparseMaps.append(sparseMap.copy())
                # get the edges information from the props file
                sparseEdges = propTitles[2:].astype(int)

                # np.save(os.path.join(saveDirectory,'sparse.st.npy'),frameChanges)

            # Gets the retinal classification metadata.
            if propTitles[0] == "Retinal":
                # Calculates the end of the final frame.
                retinal_et = np.append(
                    frameChanges[1::],
                    frameChanges[-1] + (frameChanges[14] - frameChanges[13]),
                )
                # Gets the stimulus types (assumes 1st stim is On, 2nd off, etc).
                retinal_stimType = np.empty(
                    (len(frameChanges), 1), dtype=object
                )
                # retinal_stimType[::13] = "Off"
                # retinal_stimType[1::13] = "On"
                # retinal_stimType[2::13] = "Off"
                # retinal_stimType[3::13] = "Grey"
                # retinal_stimType[4::13] = "ChirpF"
                # retinal_stimType[5::13] = "Grey"
                # retinal_stimType[6::13] = "ChirpC"
                # retinal_stimType[7::13] = "Grey"
                # retinal_stimType[8::13] = "Off"
                # retinal_stimType[9::13] = "Blue"
                # retinal_stimType[10::13] = "Off"
                # retinal_stimType[11::13] = "Green"
                # retinal_stimType[12::13] = "Off"

                retinal_stimType[12::13] = "Off"
                retinal_stimType[0::13] = "On"
                retinal_stimType[1::13] = "Off"
                retinal_stimType[2::13] = "Grey"
                retinal_stimType[3::13] = "ChirpF"
                retinal_stimType[4::13] = "Grey"
                retinal_stimType[5::13] = "ChirpC"
                retinal_stimType[6::13] = "Grey"
                retinal_stimType[7::13] = "Off"
                retinal_stimType[8::13] = "Blue"
                retinal_stimType[9::13] = "Off"
                retinal_stimType[10::13] = "Green"
                retinal_stimType[11::13] = "Off"        
                

                # Adds the data from above to the respective lists.
                retinalSt.append(frameChanges.reshape(-1, 1).copy())
                retinalEt.append(retinal_et.reshape(-1, 1).copy())
                retinalStim.append(retinal_stimType.copy())
                
            if propTitles[0] == "NaturalImages":
                stimProps = get_stimulus_info(di)
                # Gets the start times of each stimulus.
                st = frameChanges[::2].reshape(-1, 1).copy()
                # Gets the end times  of each stimulus.
                et = frameChanges[1::2].reshape(-1, 1).copy()
                naturalSt.append(st)
                naturalEt.append(et)
                naturalFiles.append(
                    stimProps.FileName.to_numpy()
                    .reshape(-1, 1)
                    .astype(str)
                    .copy()
                )
                

            # Gets the circles metadata.
            # TODO: run this in debug mode to see what exact data it gets.
            if len(propTitles) >= 3:
                if propTitles[2] == "Diameter":
                    # Gets the identity of the stimuli (see function for
                    # further details).
                    stimProps = get_stimulus_info(di)

                    # Calculates the end of the final frame.
                    circle_et = np.append(
                        frameChanges[1::],
                        frameChanges[-1] + np.median(np.diff(frameChanges)),
                    )
                    # Adds the start and end times from above to the respective
                    # lists.
                    circleSt.append(frameChanges.reshape(-1, 1).copy())
                    circleEt.append(circle_et.reshape(-1, 1).copy())

                    # Adds the data from the stimProps dictionary to the respective
                    # lists.
                    circleX.append(
                        stimProps.X.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleY.append(
                        stimProps.Y.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleDiameter.append(
                        stimProps.Diameter.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleWhite.append(
                        stimProps.White.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleDuration.append(
                        stimProps.Dur.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

            if propTitles[0] == "Ori":
                # Gets the identity of the stimuli (see function for
                # further details).
                stimProps = get_stimulus_info(di)
                # Gets the start times of each stimulus.
                st = frameChanges[::2].reshape(-1, 1).copy()
                # Gets the end times  of each stimulus.
                et = frameChanges[1::2].reshape(-1, 1).copy()

                # Checks if number of frames and stimuli match (if not, there
                # could have been an issue with the photodiode, check if there
                # are irregular frames in the photodiode trace).
                if len(stimProps) != len(st):
                    # raise ValueError(
                    #     "Number of frames and stimuli do not match. Skpping"
                    # )
                    warnings.warn("Number of frames and stimuli do not match")
                # Adds the start and end times from above to the respective
                # lists.

                gratingsSt.append(st)
                gratingsEt.append(et)
                # Adds the data from the stimProps dictionary to the respective
                # lists (all the parameters: Ori, SFreq, TFreq, Contrast).
                gratingsOri.append(
                    stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy()
                )
                gratingsSfreq.append(
                    stimProps.SFreq.to_numpy()
                    .reshape(-1, 1)
                    .astype(float)
                    .copy()
                )
                gratingsTfreq.append(
                    stimProps.TFreq.to_numpy()
                    .reshape(-1, 1)
                    .astype(float)
                    .copy()
                )
                gratingsContrast.append(
                    stimProps.Contrast.to_numpy()
                    .reshape(-1, 1)
                    .astype(float)
                    .copy()
                )
                # If a reward experiment was performed, gets the rewarded
                # stimulus data.
                if "Reward" in stimProps.columns:
                    gratingsReward.append(
                        np.array(
                            [x in "True" for x in np.array(stimProps.Reward)]
                        )
                        .reshape(-1, 1)
                        .astype(bool)
                        .copy()
                    )
                else:
                    gratingsReward.append(np.zeros_like(st) * np.nan)

        except:
            print("Error in stimulus processing in directory: " + di)
            print(traceback.format_exc())
        # Arduino data handling.
        try:

            # Gets the arduino data (see function for details).
            ardData, ardChans, at = get_arduino_data(di)
            # make sure everything is in small letters
            chans = np.array([s.lower() for s in chans])
            ardChans = np.array([s.lower() for s in ardChans])
            # Gets the sync signal form the niDaq.
            nidaqSync = nidaq[:, chans == "sync"][:, 0]
            # Gets the sync signal form the arduino.
            ardSync = ardData[:, ardChans == "sync"][:, 0]
            # Corrects the arduino time to be synched with the nidaq time
            # (see function for details).
            at_new = arduino_delay_compensation(nidaqSync, ardSync, nt, at)

            # Gets the (assumed to be) forward movement.
            movement1 = ardData[:, ardChans == "rotary1"][:, 0]
            # Gets the (assumed to be) backward movement.
            movement2 = ardData[:, ardChans == "rotary2"][:, 0]
            # Gets the wheel velocity in cm/s and the distance travelled in cm
            # (see function for details).
            v, d = detect_wheel_move(movement1, movement2, at_new)
            # Adds the wheel times to the wheelTimes list.
            wheelTimes.append(at_new + lastFrame)
            # Adds the velocity to the velocity list.
            velocity.append(v)

            # Gets the (assumed to be) face camera data.
            camera1 = ardData[:, ardChans == "camera1"][:, 0]
            # Gets the (assumed to be) body camera data.
            camera2 = ardData[:, ardChans == "camera2"][:, 0]
            # Assigns frame times to the face camera.
            # cam1Frames = assign_frame_time(camera1, fs=1, plot=False)
            # # Assigns frame times to the body camera.
            # cam2Frames = assign_frame_time(camera2, fs=1, plot=False)
            # # Uses the above frame times to get the corrected arduino frame
            # # times for the face camera.
            # cam1Frames = at_new[cam1Frames.astype(int)]
            # # Uses the above frame times to get the corrected arduino frame
            # # times for the body camera.
            # cam2Frames = at_new[cam2Frames.astype(int)]

            # look in log for video times
            # for some reason column names were different in sparse protocol
            if sparseNoise:
                logColNames = ["VideoFrame", "Video,[0-9]*", "NiDaq*"]
            else:
                logColNames = ["Video$", "Video,[0-9]*", "Analog*"]

            colNiTimes = get_recorded_video_times(
                di,
                logColNames,
                ["EyeVid", "BodyVid", "NI"],
            )
            cam1Frames = colNiTimes["EyeVid"].astype(float) / 1000
            cam2Frames = colNiTimes["BodyVid"].astype(float) / 1000
            # Get actual video data
            vfile = a = glob.glob(os.path.join(di, "Video*.avi"))[0]  # eye
            video1 = cv2.VideoCapture(vfile)
            vfile = a = glob.glob(os.path.join(di, "Video*.avi"))[1]  # body
            video2 = cv2.VideoCapture(vfile)
            # number of frames
            nframes1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
            nframes2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
            # add time stamp buffer for unknown frames
            addFrames1 = nframes1 - len(cam1Frames)
            addFrames2 = nframes2 - len(cam2Frames)
            cam1Frames = np.append(cam1Frames, np.ones(addFrames1) * np.nan)
            cam2Frames = np.append(cam2Frames, np.ones(addFrames2) * np.nan)

            if nframes1 > len(cam1Frames):
                c1f = np.ones(nframes1) * np.nan
                c1f[: len(cam1Frames)] = cam1Frames
                cam1Frames = c1f
            if nframes2 > len(cam1Frames):
                c2f = np.ones(nframes2) * np.nan
                c2f[: len(cam2Frames)] = cam2Frames
                cam2Frames = c2f

            # Adds the face times to the faceTimes list.
            faceTimes.append(cam1Frames + lastFrame)
            # Adds the body times to the bodyTimes list.
            bodyTimes.append(cam2Frames + lastFrame)
        except:
            print("Error in arduino processing in directory: " + di)
            print(traceback.format_exc())

        # Gets the last frame from the previous experiment.
        # This is then added to all the different times so the times for the
        # full session are continuous.
        lastFrame = nt[-1] + lastFrame

    # Below chunk of code saves all the metadata into separate npy files.
    np.save(
        os.path.join(saveDirectory, "calcium.timestamps.npy"),
        np.hstack(frameTimes).reshape(-1, 1),
    )
    np.save(
        os.path.join(saveDirectory, "planes.delay.npy"),
        planeTimeDelta.reshape(-1, 1),
    )

    if len(sparseMaps) > 0:
        np.save(
            os.path.join(saveDirectory, "sparse.map.npy"),
            np.vstack(sparseMaps),
        )
        np.save(
            os.path.join(saveDirectory, "sparse.startTime.npy"), np.vstack(sparseSt)
        )
        np.save(
            os.path.join(saveDirectory, "sparse.endTime.npy"), np.vstack(sparseEt)
        )
        np.save(os.path.join(saveDirectory, "sparseArea.edges.npy"), sparseEdges)
    if len(retinalStim) > 0:
        np.save(
            os.path.join(saveDirectory, "fullField.startTime.npy"), np.vstack(retinalSt)
        )
        np.save(
            os.path.join(saveDirectory, "fullField.endTime.npy"), np.vstack(retinalEt)
        )
        np.save(
            os.path.join(saveDirectory, "fullField.stim.npy"),
            np.vstack(retinalStim),
        )
    if len(gratingsSt) > 0:
        np.save(
            os.path.join(saveDirectory, "gratings.startTime.npy"),
            np.vstack(gratingsSt),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.endTime.npy"),
            np.vstack(gratingsEt),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.direction.npy"),
            np.vstack(gratingsOri),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.spatialF.npy"),
            np.vstack(gratingsSfreq),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.temporalF.npy"),
            np.vstack(gratingsTfreq),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.contrast.npy"),
            np.vstack(gratingsContrast),
        )
        
    if len(naturalSt) > 0:
        np.save(
            os.path.join(saveDirectory, "natural.startTime.npy"),
            np.vstack(naturalSt),
        )
        np.save(
            os.path.join(saveDirectory, "natural.endTime.npy"),
            np.vstack(naturalEt),
        )
        np.save(
            os.path.join(saveDirectory, "natural.fileNames.npy"),
            np.vstack(naturalFiles),
        )        

    if len(circleSt) > 0:
        np.save(
            os.path.join(saveDirectory, "circles.startTime.npy"),
            np.vstack(circleSt),
        )
        np.save(
            os.path.join(saveDirectory, "circles.endTime.npy"),
            np.vstack(circleEt),
        )
        np.save(
            os.path.join(saveDirectory, "circles.x.npy"),
            np.vstack(circleX),
        )
        np.save(
            os.path.join(saveDirectory, "circles.y.npy"),
            np.vstack(circleY),
        )
        np.save(
            os.path.join(saveDirectory, "circles.diameter.npy"),
            np.vstack(circleDiameter),
        )
        np.save(
            os.path.join(saveDirectory, "circles.isWhite.npy"),
            np.vstack(circleWhite),
        )
        np.save(
            os.path.join(saveDirectory, "circles.duration.npy"),
            np.vstack(circleDuration),
        )

    if len(gratingsReward) > 0:
        np.save(
            os.path.join(saveDirectory, "gratings.reward.npy"),
            np.vstack(gratingsReward),
        )
    if len(wheelTimes) > 0:
        np.save(
            os.path.join(saveDirectory, "wheel.timestamps.npy"),
            np.hstack(wheelTimes).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "wheel.velocity.npy"),
            np.hstack(velocity).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "eye.timestamps.npy"),
            np.hstack(faceTimes).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "body.timestamps.npy"),
            np.hstack(bodyTimes).reshape(-1, 1),
        )


def read_csv_produce_directories(dataEntry, s2pDir, zstackDir, metadataDir):
    """
    Gets all the base directories (suite2p, z Stack, metadata, save directory)
    and composes these directories for each experiment.


    Parameters
    ----------
    dataEntry : pandas DataFrame [amount of experiments, 6]
        The data from the preprocess.csv file in a pandas dataframe.
        This should have been created in the main_preprocess file; assumes
        these columns are included:
            - Name
            - Date
            - Zstack
            - IgnorePlanes
            - SaveDir
            - Process
    s2pDir : string
        Filepath to the Suite2P processed folder. For more details on what this
        should contain please look at the define_directories function
        definition in user_defs.
    zstackDir : string
        Filepath to the Z stack.For more details on what this should contain
        please look at the define_directories function definition in
        user_defs.
    metadataDir : string
        Filepath to the metadata directory.For more details on what this
        should contain please look at the define_directories function
        definition in user_defs.

    Returns
    -------
    s2pDirectory : string [s2pr\Animal\Date\suite2p]
        The concatenated Suite2P directory.
    zstackPath : string [zstackDir\Animal\Date\Z stack value from
        dataEntry\Z_stack_file.tif]
        The concatenated Z stack directory.
    metadataDirectory : string [metadataDir\Animal\Date]
        The concatenated metadata directory.
    saveDirectory : string [SaveDir from dataEntry or ]
        The save directory where all the processed files are saved. If not
        specified, will be saved in the suite2p folder.

    """
    # The data from each  dataEntry column is placed into variables.
    name = dataEntry.Name
    date = dataEntry.Date
    zstack = dataEntry.Zstack
    ignorePlanes = np.fromstring(str(dataEntry.IgnorePlanes), sep=",")
    saveDirectory = dataEntry.SaveDir
    process = dataEntry.Process

    # Joins suite2p directory with the name and the date.
    s2pDirectory = os.path.join(s2pDir, name, date, "suite2p")

    # If this path doesn't exist, returns a ValueError.
    if not os.path.exists(s2pDirectory):
        raise ValueError(
            "suite 2p directory " + s2pDirectory + "was not found."
        )
    # Checks if zStack directory number has the right shape (is not a float
    # or a NaN).
    if (type(zstack) is float) and (np.isnan(zstack)):
        zstackPath = None
        zstackDirectory = None
    else:
        # Creates the Z Stack directory.
        zstackDirectory = os.path.join(zstackDir, name, date, str(zstack))
        try:
            # Returns a path to the tif file with the Z stack within the
            # specified zstackDirectory.
            zstackPath = glob.glob(os.path.join(zstackDirectory, "*.tif"))[0]
        except:
            # If no Z stack directory was specified in the preprocess file,
            # returns a ValueError.
            # Note: the Z stack is essential for performing the Z correction!
            raise ValueError(
                "Z stack Directory not found. Please check the number in the processing csv"
            )
    # Joins suite2p directory with the name and the date.
    metadataDirectory = os.path.join(metadataDir, name, date)

    # If metadata directory does not exist, returns this ValueError.
    if not os.path.exists(metadataDirectory):
        raise ValueError(
            "metadata directory " + metadataDirectory + "was not found."
        )

    if not type(saveDirectory) is str:
        # If the saveDirectory is not a string, saves files created here
        # in a folder called PreProcessedFiles. This exists in the s2pDirectory.
        # This also means this folder is not created if the saveDirectory
        # is specified.
        saveDirectory = os.path.join(s2pDirectory, "PreprocessedFiles")
    # Creates the folder Preprocessedfiles if it doesn't exist yet
    if not os.path.isdir(saveDirectory):
        os.makedirs(saveDirectory)
    return s2pDirectory, zstackPath, metadataDirectory, saveDirectory


def read_directory_dictionary(dataEntry, s2pDirectory):
    """
    Gets all the base directories (suite2p, z Stack, metadata, save directory)
    and composes these directories for each experiment.


    Parameters
    ----------
    dataEntry : pandas DataFrame [amount of experiments, 6]
        The data from the preprocess.csv file in a pandas dataframe.
        This should have been created in the main_preprocess file; assumes
        these columns are included:
            - Name
            - Date
            - Zstack
            - IgnorePlanes
            - SaveDir
            - Process
    s2pDir : string
        Filepath to the Suite2P processed folder. For more details on what this
        should contain please look at the define_directories function
        definition in user_defs.
    zstackDir : string
        Filepath to the Z stack.For more details on what this should contain
        please look at the define_directories function definition in
        user_defs.
    metadataDir : string
        Filepath to the metadata directory.For more details on what this
        should contain please look at the define_directories function
        definition in user_defs.

    Returns
    -------
    s2pDirectory : string [s2pr\Animal\Date\suite2p]
        The concatenated Suite2P directory.
    zstackPath : string [zstackDir\Animal\Date\Z stack value from
        dataEntry\Z_stack_file.tif]
        The concatenated Z stack directory.
    metadataDirectory : string [metadataDir\Animal\Date]
        The concatenated metadata directory.
    saveDirectory : string [SaveDir from dataEntry or ]
        The save directory where all the processed files are saved. If not
        specified, will be saved in the suite2p folder.

    """
    # The data from each  dataEntry column is placed into variables.
    name = dataEntry.Name
    date = dataEntry.Date
    experiments = np.atleast_1d(dataEntry.Experiments)

    # Joins suite2p directory with the name and the date.
    s2pDirectory = os.path.join(s2pDirectory, name, date)

    # If this path doesn't exist, returns a ValueError.
    if not os.path.exists(s2pDirectory):
        raise ValueError(
            "suite 2p directory " + s2pDirectory + "was not found."
        )

    dataPaths = [s2pDirectory]
    for i, e in enumerate(experiments):
        dataPaths.append(os.path.join(s2pDirectory, str(e)))

    return dataPaths
