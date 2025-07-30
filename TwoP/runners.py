from matplotlib.colors import ListedColormap

from suite2p.registration.zalign import compute_zpos
from joblib import Parallel, delayed
# import numpy as np
import time
import traceback
import io
import os
import cv2
import skimage.io
import glob
import pickle
import scipy as sp
import tifffile
import re
import warnings
from TwoP.process_tiff import *
from TwoP.preprocess_traces import *
from Bonsai.extract_data import *
from Bonsai.behaviour_protocol_functions import *
from TwoP.general import *
from user_defs import create_2p_processing_ops, directories_to_register
import matplotlib.gridspec as gridspec


def _process_s2p_singlePlane(
        pops, currDir, zstack_raw_path, saveDirectory, piezo, plane
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
    zstack_raw_path : str [zStackDir\Animal\Z stack folder\Z stack.tif]
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
    # TODO (SS): what happens if None is returned?
    if not (os.path.exists(os.path.join(currDir, "F.npy"))):
        return None
    processed_path = os.path.join(saveDirectory, "2P_processed")
    os.makedirs(processed_path, exist_ok=True)

    # TODO: we could add option to re-register each frame to the best matching slice in the Z stack, rather than
    #  trusting the registration to the template image. BUT: zstack was collected at quite different time point and may
    #  look quite different. Also, only 10 repeats of the same plane were collected. -> excuse not to do it

    # Load neural data.
    F = np.load(os.path.join(currDir, "F.npy"), allow_pickle=True).T # (t, nROIs)
    N = np.load(os.path.join(currDir, "Fneu.npy")).T # (t, nROIs)
    isCell = np.load(os.path.join(currDir, "iscell.npy")) # (nROIs, 2)
    stat = np.load(os.path.join(currDir, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(currDir, "ops.npy"), allow_pickle=True).item()
    # Only include ROIs curated cells.
    F = F[:, isCell[:, 0].astype(bool)]
    N = N[:, isCell[:, 0].astype(bool)]
    stat = stat[isCell[:, 0].astype(bool)]
    # Remove bad frames.
    badFrames = ops['badframes']
    F[badFrames, :] = np.nan
    N[badFrames, :] = np.nan

    # TODO (SS): always specify absolute zero!
    # Add value at absolute zero (dark signal) to the traces.
    if pops["absZero"] is None:
        F = zero_signal(F)
        N = zero_signal(N)
    else:
        F = zero_signal(F, pops["absZero"])
        N = zero_signal(N, pops["absZero"])

    # If Z stack was generated (and path to file provided), perform Z correction.
    if not (zstack_raw_path is None):
        channel = ops["align_by_chan"] # imaging channel used for alignment
        zstack_path = os.path.join(
            processed_path, f"zstack_plane{plane}_chan{channel}.tif"
        )
        # Initiate parameters for Z stack alignment and reslicing. Specifies path to the registered movie (.bin) and
        # non-rigid registration method.
        ops_zcorr = ops.copy()
        ops_zcorr["reg_file"] = os.path.join(currDir, "data.bin")
        # ops_zcorr["ops_path"] = os.path.join(currDir, "ops.npy")
        ops_zcorr['nonrigid'] = False

        # Use reference image from the channel that was used for alignment.
        if channel == 1:
            refImg = ops_zcorr["meanImg"]
        else:
            refImg = ops_zcorr["meanImg_chan2"]
        # If frames were aligned using the non-functional channel (2), we also need a registered Z stack for that
        # channel to register each frame in depth.
        if channel != 1:
            zstack_functional_path = os.path.join(
                processed_path, f"zstack_plane{plane}_chan1.tif"
            )
            if not os.path.exists(zstack_functional_path):
                zstack_functional = register_zstack(
                    zstack_raw_path,
                    ops_zcorr,
                    spacing=1,
                    piezo=np.vstack((piezo[:, plane:plane + 1],
                                     np.reshape(piezo[0:1, plane + 1 % piezo.shape[1]],(1, 1)))),
                    target_image=ops['meanImg'],
                    channel=1,
                )
                # Save registered Z stack in the specified or default saveDir.
                skimage.io.imsave(zstack_functional_path, zstack_functional)
            else:
                zstack_functional = skimage.io.imread(zstack_functional_path)
        # If Z stack not saved yet, register and reslice raw Z Stack and determine correlations with movie frames.
        if not (os.path.exists(zstack_path)):
            # TODO (SS): spacing should be parameter in user_defs.py
            (zstack, best_plane) = register_zstack(
                zstack_raw_path,
                ops_zcorr,
                spacing=1,
                piezo=np.vstack((piezo[:, plane:plane + 1],
                                 np.reshape(piezo[0:1, plane + 1 % piezo.shape[1]],(1, 1)))),
                target_image=refImg,
                channel=channel
            )
            skimage.io.imsave(zstack_path, zstack)
            np.save(os.path.join(processed_path, f"bestPlane_plane{plane}.npy"), best_plane)
            _, zcorr = compute_zpos(zstack, ops_zcorr, ops_zcorr['reg_file'])
            np.save(os.path.join(processed_path, f"zcorr_plane{plane}.npy"), zcorr)
        # If Z stack was created but correlations were not, compute correlations.
        elif not os.path.exists(os.path.join(processed_path, f"zcorr_plane{plane}.npy")):
            zstack = skimage.io.imread(zstack_path)
            _, zcorr = compute_zpos(zstack, ops_zcorr, ops_zcorr['reg_file'])
        # If Z stack and correlations were saved, load them.
        else:
            if channel == 1:
                zstack = skimage.io.imread(zstack_path)
            # TODO (SS): load correct plane specific zcorr
            zcorr = np.load(os.path.join(processed_path, f"zcorr_plane{plane}.npy"))

        # If we used the non-funciton channel (2) for alignment, we now need to use the registered Z stack of the
        # functional channel (1) to correct the recorded calcium traces.
        if channel != 1:
            zstack = zstack_functional
        # Determine Z profiles for each ROI and their neuropil.
        # TODO (SS): smoothing_factor should be a parameter in user_defs.py
        F_profiles, N_profiles = extract_zprofiles(currDir, zstack, smoothing_factor=0.5, abs_zero=pops["absZero"])

        # For each frame, determine which slice in the Z stack it matches best.
        # TODO (SS): sigma should be a parameter in user_defs.py
        ztrace = (np.nanargmax(sp.ndimage.gaussian_filter1d(
            zcorr, 2, axis=0, mode='nearest'), axis=0)).astype(int)
        # Determine best reference depth.
        if pops["zcorrect_reference"] == "first": # across first experiment
            reference_depth = np.nanmedian(ztrace[:ops['frames_per_folder'][0]]).astype(int)
        else: # across all experiments
            reference_depth = np.nanmedian(ztrace).astype(int)

        # Correct ROI and neuropil traces for z motion.
        # TODO (SS): threshold should be a parameter in user_defs.py
        F_zcorrected, N_zcorrected = correct_zmotion(F, N, F_profiles, N_profiles, ztrace, reference_depth,
                              ignore_faults=pops["remove_z_extremes"], frames_per_experiment=ops["frames_per_folder"])
    else:
        # If no Z correction is performed (for example if no Z stack was given)
        # only the uncorrected delta F over F is considered.
        F_zcorrected = F
        N_zcorrected = N
        zstack = np.nan
        zcorr = np.nan

    fs = ops["fs"]
    # Perform neuropil correction.
    F_ncorrected, _, _, _ = correct_neuropil(
        F_zcorrected,
        N_zcorrected,
        fs,
        prctl_F0=pops["f0_percentile"],
        Npil_window_F0=pops["Npil_f0_window"],
    )
    # Calculate baseline fluorescence F0.
    F0 = get_F0(
        F_ncorrected,
        fs,
        prctl_F=pops["f0_percentile"],
        window_size=pops["f0_window"],
        framesPerFolder=ops["frames_per_folder"],
    )
    # Calculates delta F oer F.
    # TODO: decide whether to divide by constant of changing F0.
    dF = get_delta_F_over_F(F_ncorrected, F0)

    # Determine location of each ROI in 3D, in units of microns. Depth is relative to the top-most position of the
    # z-actuator (piezo).
    cellLocs = np.zeros((len(stat), 3))
    # First, determine depth.
    ySpan = ops["Ly"]
    for i, s in enumerate(stat):
        # Determine relative Y position wihtin FOV.
        relYpos = s["med"][0] / ySpan
        # Determine position of actuator (piezo), which depends on relative Y position. (Scanning along X dimension is
        # so fast that we can ignore it.)
        piezoInd = int(np.round((piezo.shape[0] - 1) * relYpos))
        zPos = piezo[piezoInd, plane]
        cellLocs[i, :] = np.append(s["med"], zPos)

    # Second, convert locations in horzontal plane from pixels to microns.
    # TODO (SS): Put this into separate function, so it can be adapted more easily by user. Also add parameter to decide
    #  whether to convert or not.
    lastFile = ops['filelist'][-1]
    tif = tifffile.TiffFile(lastFile)
    customTifData = tif.pages[0].tags['Artist'].value
    zoomFactor = int(re.findall('"scanZoomFactor": ([0-9])', customTifData)[0])
    # TODO (SS): Define these values as user specific inputs.
    zooms = [1, 1.5, 2, 4, 8, 16]
    totalSize = [730, 490, 360, 180, 95, 50]
    width = tif.pages[0].tags['ImageWidth'].value
    length = tif.pages[0].tags['ImageLength'].value
    zoomF = sp.interpolate.interp1d(zooms, totalSize)
    currentTotalSize = zoomF(zoomFactor)
    cellLocs[:, 0] = (cellLocs[:, 0] / length) * currentTotalSize
    cellLocs[:, 1] = (cellLocs[:, 1] / width) * currentTotalSize

    # Places all the results in a dictionary (dF/F, Z corrected dF/F,
    # z profiles, z traces and the cell locations in X, Y and Z).
    results = {
        "zCorr_stack": zcorr,
        "zTrace": ztrace,
        "zProfiles": F_profiles,
        "dff": dF,
        "locs": cellLocs,
        "cellId": np.where(isCell[0, :].astype(bool))[0],
    }

    if pops["plot"]:
        saveDirectoryPlot = os.path.join(saveDirectory, 'plots')
        os.makedirs(saveDirectoryPlot, exist_ok=True)

        # For each plane:
        # (1) ROI masks (on black background)
        # Create a colormap: black for 0, then rainbow for ROIs
        n_rois = len(stat)
        rainbow = plt.get_cmap('gist_rainbow', n_rois)
        colors = np.vstack(([0, 0, 0, 1], rainbow(np.arange(n_rois))))
        cmap = ListedColormap(colors)
        # Create a mask with ROIs numbered from 1 to n_rois.
        mask = np.zeros((ops['Ly'], ops['Lx']), dtype=np.uint8)
        for n, roi in enumerate(stat):
            ypix = roi['ypix'][~roi['overlap']]
            xpix = roi['xpix'][~roi['overlap']]
            mask[ypix, xpix] = n + 1
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(top=0.95)
        plt.imshow(mask, cmap=cmap, vmin=0, vmax=n_rois)
        # Add ROI IDs as text labels in the center of each ROI
        for n, roi in enumerate(stat):
            y, x = roi['med']
            plt.text(x, y, str(n), color='white', fontsize=12, ha='center', va='center', fontweight='bold')
        plt.tight_layout()
        plt.show()

        # (2) ROI outlines on best matching slice in stack
        # (3) Comparison between reference image and best matching slice in stack

        # TODO (SS): compare these profiles to Z profiles
        z_F_quartiles = np.ones((zstack.shape[0], F.shape[1])) * np.nan
        z_N_quartiles = np.ones((zstack.shape[0], F.shape[1])) * np.nan
        for p in np.unique(ztrace):
            z_F_quartiles[p, :] = np.percentile(F[ztrace == p, :], 25, axis=0)
            z_N_quartiles[p, :] = np.percentile(N[ztrace == p, :], 25, axis=0)

        # For each ROI:
        for i in range(dF.shape[-1]):
            # (A) Plot all data (complete time traces).
            fig = plt.figure(1, figsize=(12, 6))
            gs = gridspec.GridSpec(10, 10)
            gs.update(wspace=0.2, hspace=0.2)

            # (1) Z profile from ROI and neuropil masks, and low values of F and N traces recorded at different depths
            xtr_subplot = fig.add_subplot(gs[0:10, 0:1])
            if not (F_profiles is None) and not (ztrace is None):
                plt.plot(F_profiles[:, i], range(F_profiles.shape[0]))
                plt.legend(
                    ["Z profile"],
                    # bbox_to_anchor=(1.01, 1),
                    loc="upper left"
                )
                plt.xlabel("fluorescence")
                plt.ylabel("depth")
                plt.gca().invert_yaxis()
                plt.axhline(np.nanmedian(ztrace), c="green")
                plt.axhline(np.nanmax(ztrace), c="red")
                plt.axhline(np.nanmin(ztrace), c="blue")
                plt.text(0, np.nanmedian(ztrace), 'Median',
                         color='green', fontsize=10, va='bottom')
                plt.text(0, np.nanmax(ztrace), 'Maximum',
                         color='red', fontsize=10, va='bottom')
                plt.text(0, np.nanmin(ztrace), 'Minimum',
                         color='blue', fontsize=10, va='bottom')
                plt.xlim(0, max(F_profiles[:, i]))

            # (2) z trace of plane
            xtr_subplot = fig.add_subplot(gs[8:10, 1:10])
            if ztrace is not None:
                plt.plot(ztrace)
                plt.gca().invert_yaxis()
                plt.axhline(np.nanmedian(ztrace), c="green")
                plt.legend(
                    ["Z trace"],
                    # bbox_to_anchor=(1.01, 1),
                    loc="upper right"
                )
                plt.xlabel("time (frames)")
                plt.tick_params(axis='y', labelright=True, labelleft=False)
                plt.xlim(0, ztrace.shape[0])

            # (3) Raw and z-motion corrected ROI and neuropil traces
            xtr_subplot = fig.add_subplot(gs[0:2, 1:10])
            plt.plot(F[:, i], "b")
            plt.plot(N[:, i], "r")
            plt.legend(
                ["Fluorescence", "Neuropil"],
                # bbox_to_anchor=(1.01, 1),
                loc="upper right",
            )
            plt.xticks([])
            plt.tick_params(axis='y', labelright=True, labelleft=False)
            plt.xlim(0, dF.shape[0])

            # (4) Neuropil-corrected ROI traces and F0
            xtr_subplot = fig.add_subplot(gs[2:4, 1:10])
            plt.plot(F_ncorrected[:, i], "k")
            plt.plot(F0[:, i], "b", linewidth=4, zorder=10)
            plt.legend(
                ["Corrected F", "F0"],
                # bbox_to_anchor=(1.01, 1),
                loc="upper right",
            )
            plt.xticks([])
            plt.tick_params(axis='y', labelright=True, labelleft=False)
            plt.xlim(0, dF.shape[0])

            # (5) dF/F
            xtr_subplot_df = fig.add_subplot(gs[4:6, 1:10])
            plt.plot(dF[:, i], "b", linewidth=3)
            plt.legend(
                ["dF/F"],
                # bbox_to_anchor=(1.01, 1),
                loc="upper right",
            )
            plt.xticks([])
            plt.tick_params(axis='y', labelright=True, labelleft=False)
            plt.xlim(0, dF.shape[0])

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.savefig(
                os.path.join(
                    saveDirectoryPlot,
                    "Plane" + str(plane) + "Neuron" + str(i) + ".png",
                ),
                format="png",
            )

            with open(
                    os.path.join(
                        saveDirectoryPlot,
                        "Plane" + str(plane) + "Neuron" + str(i) + ".fig.pickle",
                    ),
                    "wb",
            ) as file:
                pickle.dump(fig, file)
            # Print Part
            # f, ax = plt.subplot_mosaic(plotArrangement)
            # ax["f"].plot(F[1:500, i], "b")
            # ax["f"].plot(N[1:500, i], "r")
            # ax["f"].legend(
            #     ["Fluorescence", "Neuropil"],
            #     bbox_to_anchor=(1.01, 1),
            #     loc="upper left",
            # )
            # ax["corr"].plot(F_ncorrected[1:500, i], "k")
            # ax["corr"].plot(F0[1:500, i], "b", linewidth=4)
            # ax["corr"].legend(
            #     ["Corrected F", "F0"],
            #     bbox_to_anchor=(1.01, 1),
            #     loc="upper left",
            # )
            # ax["zcorr"].plot(dF[1:500, i], "b", linewidth=3)
            # ax["zcorr"].plot(Fcz[1:500, i], "k", alpha=0.3)
            # ax["zcorr"].legend(
            #     ["dF/F", "dF/F z-zcorrected"],
            #     bbox_to_anchor=(1.01, 1),
            #     loc="upper left",
            # )
            # ax["zcorr"].set_xlabel("time (frames)")
            # if not zTrace is None:
            #     ax["trace"].plot(zTrace[1:500])
            #     ax["trace"].legend(
            #         ["Z trace"], bbox_to_anchor=(1.01, 1), loc="upper left"
            #     )
            # if not zprofiles is None:
            #     ax["profile"].plot(zprofiles[:, i], range(zprofiles.shape[0]))
            #     ax["profile"].legend(
            #         ["Z profile"], bbox_to_anchor=(1.01, 1), loc="upper left"
            #     )
            #     ax["profile"].set_xlabel("fluorescence")
            #     ax["profile"].set_xlabel("depth")

            # (B) Plot traces in a zoomed-in view (first 500 frames).
            plt.close()
            fig = plt.figure(1, figsize=(12, 6))
            gs = gridspec.GridSpec(10, 10)
            gs.update(wspace=0.2, hspace=0.2)

            # (1) z trace of plane

            # (2) Raw and z-motion corrected ROI and neuropil traces
            xtr_subplot = fig.add_subplot(gs[0:2, 1:10])
            plt.plot(F[:500, i], "b")
            plt.plot(N[:500, i], "r")
            plt.legend(
                ["Fluorescence", "Neuropil"],
                loc="upper right",
            )
            plt.xticks([])
            plt.tick_params(axis='y', labelright=True, labelleft=False)
            plt.xlim(0, dF[:500].shape[0])

            # (4) Neuropil-corrected ROI traces and F0
            xtr_subplot = fig.add_subplot(gs[2:4, 1:10])
            plt.plot(F_ncorrected[:500, i], "k")
            plt.plot(F0[:500, i], "b", linewidth=4, zorder=10)
            plt.legend(
                ["Corrected F", "F0"],
                loc="upper right",
            )
            plt.xticks([])
            plt.tick_params(axis='y', labelright=True, labelleft=False)
            plt.xlim(0, dF[:500].shape[0])

            # (5) dF/F
            xtr_subplot_df = fig.add_subplot(gs[4:6, 1:10])
            plt.plot(dF[:500, i], "b", linewidth=3)
            plt.legend(
                ["dF/F"],
                loc="upper right",
            )
            plt.xticks([])
            plt.tick_params(axis='y', labelright=True, labelleft=False)
            plt.xlim(0, dF[:500].shape[0])

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            plt.savefig(
                os.path.join(
                    saveDirectoryPlot,
                    "Plane" + str(plane) + "Neuron" + str(i) + "_zoom.png",
                ),
                format="png",
            )

            with open(
                    os.path.join(
                        saveDirectoryPlot,
                        "Plane"
                        + str(plane)
                        + "Neuron"
                        + str(i)
                        + "_zoom.fig.pickle",
                    ),
                    "wb",
            ) as file:
                pickle.dump(fig, file)

            plt.close("all")
    return results


def process_s2p_directory(
        suite2p_directory,
        pops=create_2p_processing_ops(),
        piezo=None,
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
    suite2p_directory : str [s2pDir/Animal/Date/suite2p]
        the suite2p parent directory, where the plane directories are.
    piezo : [time X plane] um
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
        saveDirectory = os.path.join(suite2p_directory, "ProcessedData")
    if not os.path.isdir(saveDirectory):
        os.makedirs(saveDirectory)
    # Creates a list which contains the directories to the subfolders for each
    # plane.
    planeDirs = glob.glob(os.path.join(suite2p_directory, "plane*"))
    planeDirs = np.sort(planeDirs)
    # Loads the ops dictionary from the combined directory.
    ops = np.load(
        os.path.join(planeDirs[-1], "ops.npy"), allow_pickle=True
    ).item()

    isBoutons = ('selected_plane' in ops.keys())
    # Loads the number of planes into a variable.
    # Creates an array with the plane range.
    planeRange = [int(re.findall(r'plane(\d+)', s)[0]) for s in planeDirs]
    # TODO (SS) OLD: planeRange = np.arange(len(planeDirs))
    # Removes the ignored plane (if specified) from the plane range array.
    if isBoutons or ignorePlanes is None:
        ignorePlanes = []
    # TODO (SS) OLD:
    # else:
    #     ignorePlanes = [i for i, s in enumerate(planeDirs) if int(
    #         re.findall(r'plane|\d+', s)[-1]) in ignorePlanes]

    planeRange = np.delete(planeRange, ignorePlanes)
    # TODO (SS) OLD:
    # # find what plane directories exist and match piezo plane number to them
    # # TODO (SS): Consider not deleting any planes from piezoTraces
    # if not (len(ignorePlanes) == 0):
    #     # TODO (SS): if planeDirs does not contain plane0 but ignorePlanes does, the 1st plane will be ignored
    #     planeRange = np.delete(planeRange, ignorePlanes)
    #     repPlanes = [int(re.findall(r'plane|\d+', s)[-1]) for s in planeDirs]
    #     piezoTraces = piezoTraces[:, repPlanes] if not (
    #         isBoutons) else piezoTraces[:, ops['selected_plane']]

    # Determine the absolute time before processing.
    preTime = time.time()

    # Specifies the amount of parallel jobs to decrease processing time.
    # If in debug mode, there will be no parallel processing.
    if not debug:
        jobnum = 4
    else:
        jobnum = 1

    # not a bouton recording proceed normally
    if not isBoutons:
        # Processes the 2P data for the planes specified in the plane range.
        # This gives a list of dictionaries with all the planes.
        # Refer to the function for a more thorough description.
        # results = Parallel(n_jobs=jobnum, verbose=5)(
        #     delayed(_process_s2p_singlePlane)(
        #         pops, planeDirs[plane], zstackPath, saveDirectory, piezo, plane
        #     )
        #     for plane in planeRange
        # )
        results = [
            _process_s2p_singlePlane(
                pops, planeDirs[plane], zstackPath, saveDirectory, piezo, plane
            )
            for plane in planeRange
        ]
        # signalList = _process_s2p_singlePlane(planeDirs,zstackPath,saveDirectory,piezoTraces[:,0],1)
        # Determines the absolute time after processing.
    # bouton recording
    else:
        p = ops['selected_plane']
        results = Parallel(n_jobs=jobnum, verbose=5)(
            delayed(_process_s2p_singlePlane)(
                pops, list([planeDirs[-1]]), zstackPath, saveDirectory, piezo[:, p].reshape(-1, 1), p
            )
            for p in [0]
        )

    # If plotting: ztraces of all planes.

    postTime = time.time()
    print("Processing took: " + str(postTime - preTime) + " ms")

    # TODO: only create planes, the rest is already in results.
    # Collect results from all the planes.
    planes = np.array([])
    signalList = []
    signalLocs = []
    zTraces = []
    zProfiles = []
    zCorrs = []
    cellIds = []
    for i in range(len(results)):
        if not (results[i] is None):
            zCorrs.append(results[i]["zCorr_stack"])
            zTraces.append(results[i]["zTrace"])
            zProfiles.append(results[i]["zProfiles"])
            signalList.append(results[i]["dff"])
            signalLocs.append(results[i]["locs"])
            cellIds.append(results[i]["cellId"])
            res = signalList[i]
            planes = np.append(planes, np.ones(res.shape[1]) * planeRange[i]) if not isBoutons else np.append(
                planes, np.ones(res.shape[1]) * ops['selected_plane'])
    # Clip all signals the same length (to the shortest signal).
    minLength = np.inf
    for i in range(len(signalList)):
        minLength = np.min((signalList[i].shape[0], minLength))
    for i in range(len(signalList)):
        signalList[i] = signalList[i][: int(minLength), :]
        if not zTraces[i] is None:
            zTraces[i] = zTraces[i][: int(minLength)]
            zCorrs[i] = zCorrs[i][: int(minLength)]
    # TODO: Do this when saving.
    # Combine all results into arrays.
    signals = np.hstack(signalList)
    locs = np.vstack(signalLocs)
    zProfile = np.hstack(zProfiles)
    zTrace = np.vstack(zTraces)
    zCorrs = np.swapaxes(np.dstack(zCorrs).T, 1, 2)
    cellIds = np.hstack(cellIds)

    # TODO: change naming.
    # Saves the results as individual npy files.
    np.save(os.path.join(saveDirectory, "calcium.dff.npy"), signals)
    np.save(os.path.join(saveDirectory, "rois.planes.npy"), planes)
    np.save(os.path.join(saveDirectory, "rois.id.npy"), cellIds)
    np.save(os.path.join(saveDirectory, "rois.xyz.npy"), locs)
    np.save(os.path.join(saveDirectory, "rois.zProfiles.npy"), zProfile.T)
    np.save(os.path.join(saveDirectory, "planes.zTrace"), zTrace)
    np.save(os.path.join(saveDirectory, "planes.zcorrelation"), zCorrs)


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

    # if saveDirectory is None:
    #     saveDirectory = os.path.join(suite2pDirectory, "ProcessedData")
    # metadataDirectory_dirList = glob.glob(os.path.join(metadataDirectory,'*'))

    if not os.path.isdir(saveDirectory):
        os.makedirs(saveDirectory)

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

    # lick spout data
    licks = []
    lickTimes = []

    stimulusProps = []
    stimulusTypes = []

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
            print(traceback.format_exc())

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

        if ("Spont" in propTitles[0]) | (len(sparseFile) != 0):
            sparseNoise = True

        try:
            # Gets the photodiode data.
            photodiode = nidaq[:, chans == "photodiode"]
            # Gets the frames where photodiode changes are detected.
            frameChanges = detect_photodiode_changes(
                photodiode, plot=pops["plot"]
            )
            frameChanges += lastFrame
            if (len(frameChanges) == 0):
                raise Exception("No Frames")

        except:
            print("Error in frame time extraction in directory: " + di)
            print("\nresorting to giving first and last frame")
            print(traceback.format_exc())
            frameChanges = [lastFrame, nt[-1] + lastFrame]
        try:
            # process stimuli
            stimulusResults = process_stimulus(propTitles, di, frameChanges)
            stimulusProps.append(stimulusResults)
            stimulusTypes.append(propTitles[0][0])
        except:
            print("Error in stimulus processing in directory: " + di)
            print(traceback.format_exc())

        try:
            # lick spout
            lickSpout = np.ones_like(frameclock) * np.nan
            if ("lick" in chans):
                lickSpout = nidaq[:, chans == "lick"]
            licks.append(lickSpout)
            lickTimes.append(nt + lastFrame)
        except:
            print("Could not load licking data")
        # get video data if possible
        try:
            nframes1 = np.nan
            nframes2 = np.nan
            # Get actual video data
            vfile = glob.glob(os.path.join(
                di, "Video[0-9]*.avi"))  # eye
            if (len(vfile) > 0):
                vfile = vfile[0]
                video1 = cv2.VideoCapture(vfile)
                nframes1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
            vfile = glob.glob(os.path.join(
                di, "Video[a-zA-Z]*.avi"))  # body
            if (len(vfile) > 0):
                vfile = vfile[0]
                video2 = cv2.VideoCapture(vfile)
                nframes2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
            # number of frames

        except:
            print("Error in loading video file in directory: " + di)
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

            try:
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
            except:
                print("Error in wheel processing in directory: " + di)
                print(traceback.format_exc())

            try:
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
                vfile = glob.glob(os.path.join(
                    di, "Video[0-9]*.avi"))[0]  # eye
                video1 = cv2.VideoCapture(vfile)
                vfile = glob.glob(os.path.join(
                    di, "Video[a-zA-Z]*.avi"))[0]  # body
                video2 = cv2.VideoCapture(vfile)
                # number of frames
                nframes1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
                nframes2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
                # add time stamp buffer for unknown frames
                addFrames1 = nframes1 - len(cam1Frames)
                addFrames2 = nframes2 - len(cam2Frames)

                if nframes1 > len(cam1Frames):
                    c1f = np.ones(nframes1) * np.nan
                    c1f[: len(cam1Frames)] = cam1Frames
                    cam1Frames = c1f
                if nframes2 > len(cam2Frames):
                    c2f = np.ones(nframes2) * np.nan
                    c2f[: len(cam2Frames)] = cam2Frames
                    cam2Frames = c2f

                # Adds the face times to the faceTimes list.
                faceTimes.append(cam1Frames + lastFrame)
                # Adds the body times to the bodyTimes list.
                bodyTimes.append(cam2Frames + lastFrame)
            except:
                print("Error in camera processing in directory: " + di)
                print(traceback.format_exc())
                # make a nan array with length of video (cannot place the time)
                print('filling the possible times with empty timestamps')
                if ~np.isnan(nframes1):
                    faceTimes.append(np.ones(nframes1) * np.nan)
                if ~np.isnan(nframes2):
                    bodyTimes.append(np.ones(nframes2) * np.nan)

        except:
            print("Error in arduino processing in directory: " + di)
            print(traceback.format_exc())

            print('filling the possible times with empty timestamps')
            if ~np.isnan(nframes1):
                faceTimes.append(np.ones(nframes1) * np.nan)
            if ~np.isnan(nframes2):
                bodyTimes.append(np.ones(nframes2) * np.nan)

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

    # concatante stimuli and save them
    save_stimuli(saveDirectory, stimulusTypes, stimulusProps)

    if len(wheelTimes) > 0:
        np.save(
            os.path.join(saveDirectory, "wheel.timestamps.npy"),
            np.hstack(wheelTimes).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "wheel.velocity.npy"),
            np.hstack(velocity).reshape(-1, 1),
        )
    if (len(faceTimes) > 0):
        np.save(
            os.path.join(saveDirectory, "eye.timestamps.npy"),
            np.hstack(faceTimes).reshape(-1, 1),
        )
    if (len(bodyTimes) > 0):
        np.save(
            os.path.join(saveDirectory, "body.timestamps.npy"),
            np.hstack(bodyTimes).reshape(-1, 1),
        )
    if (len(licks) > 0):
        np.save(
            os.path.join(saveDirectory, "spout.timestamps.npy"),
            np.hstack(lickTimes).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "spout.licks.npy"),
            np.vstack(licks).reshape(-1, 1),
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
    saveDirectory = dataEntry.SaveDir
    process = dataEntry.Process

    # Joins suite2p directory with the name and the date.
    s2pDirectory = os.path.join(s2pDir, name, date, "suite2p")
    saveDirectory = os.path.join(saveDirectory, name, date)

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
    # Joins metadata directory with the name and the date.
    metadataDirectory = os.path.join(metadataDir, name, date)

    # If metadata directory does not exist, returns this ValueError.
    if not os.path.exists(metadataDirectory):
        raise ValueError(
            "metadata directory " + metadataDirectory + "was not found."
        )

    # Creates save directory if it doesn't exist yet
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
