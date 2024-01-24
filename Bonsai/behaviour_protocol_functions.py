# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:51:58 2024

@author: liad0
"""
import numpy as np
from Data.Bonsai.log_extraction_functions import get_stimulus_info, get_sparse_noise
import glob
import os
import warnings


def stimulus_circles(directory, frameChanges):

    stimProps = get_stimulus_info(directory)
    # Calculates the end of the final frame.
    circle_et = np.append(
        frameChanges[1::],
        frameChanges[-1] + np.median(np.diff(frameChanges)),
    )

    circle_st = frameChanges

    return {'circles.startTime.npy': circle_st.reshape(-1, 1).copy(),
            'circles.endTime.npy': circle_et.reshape(-1, 1).copy(),
            'circles.x.npy': stimProps.X.to_numpy().reshape(-1, 1).astype(float).copy(),
            'circles.y.npy': stimProps.Y.to_numpy()
            .reshape(-1, 1)
            .astype(float)

            .copy(), 'circles.diameter.npy': stimProps.Diameter.to_numpy()
            .reshape(-1, 1)
            .astype(float)
            .copy(), 'circles.isWhite.npy': stimProps.White.to_numpy()
            .reshape(-1, 1)
            .astype(float)
            .copy(), 'circles.duration.npy': stimProps.Dur.to_numpy()
            .reshape(-1, 1)
            .astype(float)
            .copy(),
            "circlesExp.intervals": [circle_st[0], circle_et[-1]]
            }


def stimulus_classification(directory, frameChanges):
    retinal_et = np.append(
        frameChanges[1::],
        frameChanges[-1] + (frameChanges[14] - frameChanges[13]),
    )

    retinal_st = frameChanges

    retinal_stimType = np.empty(
        (len(frameChanges), 1), dtype=object
    )
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

    return {'fullField.startTime.npy': retinal_st.reshape(-1, 1).copy(),
            "fullField.endTime.npy": retinal_et.reshape(-1, 1).copy(),
            "fullField.stim.npy": retinal_stimType.copy(),
            "fullFieldExp.intervals.npy": [retinal_st[0], retinal_et[-1]]
            }


def stimulus_sparse(directory, frameChanges):
    sparseMap = get_sparse_noise(directory)
    sparseMap = sparseMap[: len(frameChanges), :, :]
    propsFile = glob.glob(os.path.join(directory, "props*.csv"))
    propTitles = np.loadtxt(
        propsFile[0], dtype=str, delimiter=",", ndmin=2
    ).T

    # Calculates the end of the final frame.
    sparse_et = np.append(
        frameChanges[1::],
        frameChanges[-1] + np.median(np.diff(frameChanges)),
    )

    sparse_st = frameChanges
    sparseEdges = propTitles[2:].astype(int)

    return {"sparse.map.npy": sparseMap.copy(),
            "sparse.startTime.npy": sparse_st.reshape(-1, 1).copy(),
            "sparse.endTime.npy": sparse_et.reshape(-1, 1).copy(),
            "sparseExp.edges.npy": sparseEdges,
            "sparseExp.intervals.npy": [sparse_st[0], sparse_et[-1]],
            }


def stimulus_gratings(directory, frameChanges):

    # Gets the identity of the stimuli (see function for
    # further details).
    stimProps = get_stimulus_info(directory)
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

    if "Reward" in stimProps.columns:
        reward = np.array(
            [x in "True" for x in np.array(stimProps.Reward)]).reshape(-1, 1).astype(bool).copy()
    else:
        reward = np.ones_like(st) * np.nan

    return {"gratings.startTime.npy": st,
            "gratings.endTime.npy": et,
            "gratings.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
            "gratings.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratings.temporalF.npy": stimProps.TFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratings.contrast.npy": stimProps.Contrast.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratings.reward.npy": reward,
            "gratingsExp.intervals.npy": [st[0], et[-1]]
            }


def stimulus_naturalImages(directory, frameChanges):
    stimProps = get_stimulus_info(directory)
    # Gets the start times of each stimulus.
    st = frameChanges[::2].reshape(-1, 1).copy()
    # Gets the end times  of each stimulus.
    et = frameChanges[1::2].reshape(-1, 1).copy()

    return {"natural.startTime.npy": st,
            "natural.endTime.npy": et,
            "natural.fileNames.npy": stimProps.FileName.to_numpy().reshape(-1, 1).astype(str).copy(),
            "naturalExp.intervals.npy": [st[0, 0], et[-1, 0]]
            }


def stimulus_spont(directory, frameChanges):
    return {"darkScreen.intervals": [frameChanges[0], frameChanges[-1]]}


stimulus_prcoessing_dictionary = {
    "Gratings": stimulus_gratings,
    "Circles": stimulus_circles,
    "Retinal": stimulus_classification,
    "NaturalImages": stimulus_naturalImages,
    "Sparse": stimulus_sparse,
    "Spont": stimulus_spont,
}
