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
import pandas as pd


def stimulus_circles(directory, frameChanges):

    stimProps = get_stimulus_info(directory)
    # Calculates the end of the final frame.
    circle_et = np.append(
        frameChanges[1::],
        frameChanges[-1] + np.median(np.diff(frameChanges)),
    )

    circle_st = frameChanges

    # make sure no spurious signal is caught
    durs = np.diff(circle_st)
    if (durs[-1] > np.max(stimProps.Dur.astype(np.float64))*10):
        circle_st = circle_st[:-1]
        circle_et = circle_et[:-1]

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
    sparseEdges = propTitles[1:].astype(int)

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
        if (len(et) == len(stimProps)):
            warnings.warn(
                "Assuming there was a false photodiode rise in the beginning, but check!")
            st = frameChanges[0:-1:2].reshape(-1, 1).copy()
            # Gets the end times  of each stimulus.
            # et = frameChanges[2::2].reshape(-1, 1).copy()
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
            "gratingsExp.intervals.npy": np.atleast_2d([st[0], et[-1]]).T,
            # TODO: check if this is usefull; should add same line to gratings
            "gratingsExp.description.npy": "Gratings",
            }


def stimulus_gratingsLuminance(directory, frameChanges):
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
        if (len(et) == len(stimProps)):
            warnings.warn(
                "Assuming there was a false photodiode rise in the beginning, but check!")
            st = frameChanges[0:-1:2].reshape(-1, 1).copy()
            # Gets the end times  of each stimulus.
            # et = frameChanges[2::2].reshape(-1, 1).copy()
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
            "gratings.luminance.npy": stimProps.Lum.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratings.reward.npy": reward,
            "gratingsExp.intervals.npy": np.atleast_2d([st[0], et[-1]]).T,
            # TODO: check if this is usefull; should add same line to gratings
            "gratingsExp.description.npy": "Gratings",
            }


def stimulus_gratings_reward(directory, frameChanges):

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
        if (len(et) == len(stimProps)):
            warnings.warn(
                "Assuming there was a false photodiode rise in the beginning, but check!")
            st = frameChanges[0:-1:2].reshape(-1, 1).copy()
            # Gets the end times  of each stimulus.
            # et = frameChanges[2::2].reshape(-1, 1).copy()
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
            "gratingsExp.intervals.npy": [st[0], et[-1]],
            # TODO: check if this is usefull; should add same line to gratings
            "gratingsExp.description.npy": "Gratings",
            }


def stimulus_naturalImages(directory, frameChanges):
    stimProps = get_stimulus_info(directory, 'FileIndex')

    # Gets the start times of each stimulus.
    # need to ignore the first stimulus since it's just the small square onset
    st = frameChanges[1::2].reshape(-1, 1).copy()
    # Gets the end times  of each stimulus.
    et = frameChanges[2::2].reshape(-1, 1).copy()

    if (len(st) > len(et)):
        if (st[-1] > et[-1]):
            st = st[:-1]

    imgList = []
    imgFile = glob.glob(os.path.join(directory, "imgList*.csv"))

    if (len(imgFile) > 0):
        imgList = pd.read_csv(imgFile[0], header=None).to_numpy()

    return {"natural.startTime.npy": st,
            "natural.endTime.npy": et,
            "natural.fileIndex.npy": stimProps.FileIndex.to_numpy().reshape(-1, 1).astype(int).copy(),
            "naturalExp.imgList.npy": {'imgList': np.array(imgList)},
            "naturalExp.intervals.npy": [st[0, 0], et[-1, 0]]
            }


def stimulus_spont(directory, frameChanges):
    return {"darkScreen.intervals": [frameChanges[0], frameChanges[-1]]}


def stimulus_spont_grey(directory, frameChanges):
    return {"greyScreen.intervals": [frameChanges[0], frameChanges[-1]]}


def stimulus_flicker(directory, frameChanges):
    # Gets the identity of the stimuli (see function for
    # further details).
    # stimProps = get_stimulus_info(directory)
    flicker_stimType = np.zeros(60)  # Asuming 10 reps per each contrast block

    flicker_stimType[0::2] = 0.05  # SD from Low contrast
    flicker_stimType[1::2] = 0.175  # SD from High contrast
    flicker_stimType = flicker_stimType.reshape(-1, 1)

    # Checks if number of frames and stimuli match (if not, there
    # could have been an issue with the photodiode, check if there
    # are irregular frames in the photodiode trace).

    if (len(flicker_stimType)+1) == len(frameChanges):

        # Gets the start times of each stimulus.
        st = frameChanges[:-1].reshape(-1, 1).copy()
        # Gets the end times  of each stimulus.
        et = frameChanges[1:].reshape(-1, 1).copy()
    else:
        warnings.warn(f'''Number of frames and stimuli do not match.
                      Assuming there was a false photodiode rise at the end, but check!''')

        st = frameChanges[0:-2].reshape(-1, 1).copy()
        et = frameChanges[1:-1].reshape(-1, 1).copy()

    return {"flicker.startTime.npy": st,
            "flicker.endTime.npy": et,
            "flicker.contrast.npy": flicker_stimType,
            "flickerExp.intervals.npy": [st[0], et[-1]],
            }


def stimulus_oddball(directory, frameChanges):
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
    if (len(stimProps)+1 == len(st)) & (len(stimProps) == len(et)):
        # raise ValueError(
        #     "Number of frames and stimuli do not match. Skpping"
        # )
        warnings.warn("Number of frames and stimuli do not match")
        if (len(et) == len(stimProps)):
            warnings.warn(
                "Assuming there was a false photodiode rise in the end, but check!")
            st = frameChanges[0:-1:2].reshape(-1, 1).copy()
            et = frameChanges[1:-1:2].reshape(-1, 1).copy()
    
    elif (len(stimProps)+1 == len(st)) & (len(stimProps)+1 == len(et)):
        warnings.warn("Number of frames and stimuli do not match")
        warnings.warn(
                "Assuming there was 2 false photodiode rise: beggining and end, but check!")
        st = frameChanges[1:-1:2].reshape(-1, 1).copy()
        et = frameChanges[2:-1:2].reshape(-1, 1).copy()

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
            "gratingsExp.intervals.npy": np.atleast_2d([st[0], et[-1]]).T,
            # TODO: check if this is usefull; should add same line to gratings
            "gratingsExp.description.npy": 'Oddball'
            }


def stimulus_gratingsStep(directory, frameChanges):
    '''
    Step increasing/decreasing TFreq
    '''
    stimProps = get_stimulus_info(directory)
    # Gets the start times of each stimulus.
    st = frameChanges[::2].reshape(-1, 1).copy()
    # Gets the end times  of each stimulus.
    et = frameChanges[1::2].reshape(-1, 1).copy()

    # Checks if number of frames and stimuli match (if not, there
    # could have been an issue with the photodiode, check if there
    # are irregular frames in the photodiode trace).
    if len(stimProps) != len(st):

        warnings.warn("Number of frames and stimuli do not match")
        if (len(et) == len(stimProps)):
            warnings.warn(
                "Assuming there was a false photodiode rise in the end, but check!")
            st = frameChanges[0:-1:2].reshape(-1, 1).copy()

    # Adds the start and end times from above to the respective
    # lists.

    return {"gratingsStep.startTime.npy": st,
            "gratingsStep.endTime.npy": et,
            "gratingsStep.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
            "gratingsStep.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsStep.startTemporalF.npy": stimProps.TFreqStart.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsStep.endTemporalF.npy": stimProps.TFreqEnd.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsStep.contrast.npy": stimProps.Contrast.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsStepExp.intervals.npy": [st[0], et[-1]],
            }


def stimulus_gratingsContrastStep(directory, frameChanges):

    stimProps = get_stimulus_info(directory)
    # Gets the start times of each stimulus.
    st = frameChanges[::2].reshape(-1, 1).copy()
    # Gets the end times  of each stimulus.
    et = frameChanges[1::2].reshape(-1, 1).copy()

    # Checks if number of frames and stimuli match (if not, there
    # could have been an issue with the photodiode, check if there
    # are irregular frames in the photodiode trace).
    if len(stimProps) != len(st):

        warnings.warn("Number of frames and stimuli do not match")
        if (len(et) == len(stimProps)):
            warnings.warn(
                "Assuming there was a false photodiode rise in the end, but check!")
            st = frameChanges[0:-1:2].reshape(-1, 1).copy()

    # Adds the start and end times from above to the respective
    # lists.

    return {"gratingsContrastStep.startTime.npy": st,
            "gratingsContrastStep.endTime.npy": et,
            "gratingsContrastStep.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
            "gratingsContrastStep.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsContrastStep.temporalF.npy": stimProps.TFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsContrastStep.contrastStart.npy": stimProps.Contrast_Start.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsContrastStep.contrastEnd.npy": stimProps.Contrast_End.to_numpy().reshape(-1, 1).astype(float).copy(),
            "gratingsContrastStep.intervals.npy": [st[0], et[-1]],
            }


def stimulus_classificationExtended(directory, frameChanges): 
    retinal_et = np.append(
        frameChanges[1::],
        frameChanges[-1] + (frameChanges[14] - frameChanges[13]),
    )

    retinal_st = frameChanges

    retinal_stimType = np.empty(
        (len(frameChanges), 1), dtype=object
    )
    retinal_stimType[18::19] = "Off"
    retinal_stimType[0::19] = "On"
    retinal_stimType[1::19] = "Off"
    retinal_stimType[2::19] = "Grey"
    retinal_stimType[3::19] = "ChirpF"
    retinal_stimType[4::19] = "Grey"
    retinal_stimType[5::19] = "ChirpCIncreasing"
    retinal_stimType[6::19] = "Grey"
    retinal_stimType[7::19] = "ChirpCDecreasing"
    retinal_stimType[8::19] = "Grey"
    retinal_stimType[9::19] = "ChirpCIncreasing2Hz"
    retinal_stimType[10::19] = "Grey"
    retinal_stimType[11::19] = "ChirpCDecreasing2Hz"   
    retinal_stimType[12::19] = "Grey"
    retinal_stimType[13::19] = "Off"
    retinal_stimType[14::19] = "Blue"
    retinal_stimType[15::19] = "Off"
    retinal_stimType[16::19] = "Green"
    retinal_stimType[17::19] = "Off"

    return {'fullFieldExtended.startTime.npy': retinal_st.reshape(-1, 1).copy(),
            "fullFieldExtended.endTime.npy": retinal_et.reshape(-1, 1).copy(),
            "fullFieldExtended.stim.npy": retinal_stimType.copy(),
            "fullFieldExtended.intervals.npy": [retinal_st[0], retinal_et[-1]]
            }
    

stimulus_prcoessing_dictionary = {
    "Gratings": stimulus_gratings,
    "GratingsReward": stimulus_gratings_reward,
    "Circles": stimulus_circles,
    "Retinal": stimulus_classification,
    "NaturalImages": stimulus_naturalImages,
    "Sparse": stimulus_sparse,
    "Spont": stimulus_spont,
    "SpontGrey": stimulus_spont_grey,
    "Flicker": stimulus_flicker,
    "Oddball": stimulus_oddball,
    "GratingsStep": stimulus_gratingsStep,
    "GratingsLuminance": stimulus_gratingsLuminance,
    "GratingsContrastStep": stimulus_gratingsContrastStep,
    "RetinalExtended": stimulus_classificationExtended,
}
