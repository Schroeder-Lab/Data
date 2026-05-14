import numpy as np

from Bonsai_TwoP import extract_data
from Bonsai_TwoP.log_extraction_functions import get_stimulus_info, get_sparse_noise
import glob
import os
import warnings
import pandas as pd


def stimulus_circles(directory):

    stimuli = get_stimulus_info(directory)

    result = {
        'circles.xPos.npy': stimuli.X.to_numpy().reshape(-1, 1).astype(float),
        'circles.yPos.npy': stimuli.Y.to_numpy().reshape(-1, 1).astype(float),
        'circles.diameter.npy': stimuli.Diameter.to_numpy().reshape(-1, 1).astype(float),
        'circles.isWhite.npy': stimuli.White.to_numpy().reshape(-1, 1).astype(float)
    }

    num_trials = len(stimuli)
    time_samples = stimuli.timestamp.to_numpy().reshape(-1, 1).astype(float)

    return result, num_trials, "circles", time_samples


def stimulus_classification(directory):
    return {}, 10 * 13, "fullField", None


def stimulus_sparse(directory):
    sparse_map = get_sparse_noise(directory)
    props_file = glob.glob(os.path.join(directory, "props*.csv"))
    prop_titles = np.loadtxt(
        props_file[0], dtype=str, delimiter=",", ndmin=2
    ).T
    # Last 4 entries: [bottom, top, left, right] where bottom and top are positive if above horizon;
    # transform to: [left, right, top, bottom]
    sparse_edges = prop_titles[-4:].astype(int)
    sparse_edges = sparse_edges[[2, 3, 1, 0]].reshape(1, -1)

    events_log = extract_data.get_log_entry(directory, ["SparseNoiseFrame"], "NiDaq*")[0]

    results = {
        "sparse.map.npy": sparse_map,
        "sparseExp.edges.npy": sparse_edges
        }

    num_trials = sparse_map.shape[0]
    time_samples = events_log["SparseNoiseFrame"].timestamp.to_numpy().reshape(-1, 1).astype(float)

    return results, num_trials, "sparse", time_samples


def stimulus_gratings(directory):
    stimuli = get_stimulus_info(directory)

    if "Reward" in stimuli.columns:
        reward = np.array([value == "True" for value in stimuli.Reward], dtype=bool).reshape(-1, 1).copy()
    else:
        reward = None

    result = {
        "gratings.direction.npy": stimuli.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
        "gratings.spatialF.npy": stimuli.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
        "gratings.temporalF.npy": stimuli.TFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
        "gratings.contrast.npy": stimuli.Contrast.to_numpy().reshape(-1, 1).astype(float).copy(),
        "gratingsExp.description.npy": "Gratings"
    }

    if reward is not None:
        result["gratings.reward.npy"] = reward

    num_trials = len(stimuli)
    time_samples = stimuli.timestamp.to_numpy().reshape(-1, 1).astype(float)

    return result, num_trials, "gratings", time_samples


def stimulus_spont(directory):
    return {}, 0, "darkScreen", None


def stimulus_spont_grey(directory):
    return {}, 0, "greyScreen", None


# def stimulus_gratingsLuminance(directory, frameChanges):
#     # Gets the identity of the stimuli (see function for
#     # further details).
#     stimProps = get_stimulus_info(directory)
#     # Gets the start times of each stimulus.
#     st = frameChanges[::2].reshape(-1, 1).copy()
#     # Gets the end times  of each stimulus.
#     et = frameChanges[1::2].reshape(-1, 1).copy()
#
#     # Checks if number of frames and stimuli match (if not, there
#     # could have been an issue with the photodiode, check if there
#     # are irregular frames in the photodiode trace).
#
#     if (len(et)<len(st)):
#         warnings.warn(
#             "Assuming the last end time was not recorded and assuming normal presentation time")
#         mDur = np.nanmedian(et-st[:-1])
#         et = np.atleast_2d(np.append(et,st[-1]+mDur)).T
#     if len(stimProps) != len(st):
#         # raise ValueError(
#         #     "Number of frames and stimuli do not match. Skpping"
#         # )
#         warnings.warn("Number of frames and stimuli do not match")
#         if (len(et) == len(stimProps)):
#             warnings.warn(
#                 "Assuming there was a false photodiode rise in the beginning, but check!")
#             st = frameChanges[0:-1:2].reshape(-1, 1).copy()
#             # Gets the end times  of each stimulus.
#             # et = frameChanges[2::2].reshape(-1, 1).copy()
#     # Adds the start and end times from above to the respective
#     # lists.
#
#     if "Reward" in stimProps.columns:
#         reward = np.array(
#             [x in "True" for x in np.array(stimProps.Reward)]).reshape(-1, 1).astype(bool).copy()
#     else:
#         reward = np.ones_like(st) * np.nan
#
#     return {"gratings.startTime.npy": st,
#             "gratings.endTime.npy": et,
#             "gratings.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
#             "gratings.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.temporalF.npy": stimProps.TFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.contrast.npy": stimProps.Contrast.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.luminance.npy": stimProps.Luminance.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.reward.npy": reward,
#             "gratingsExp.intervals.npy": np.atleast_2d([st[0], et[-1]]).T,
#             "gratingsExp.description.npy": "Gratings",
#             }

# def stimulus_Luminance(directory, frameChanges):
#     # Gets the identity of the stimuli (see function for
#     # further details).
#     stimProps = get_stimulus_info(directory,['Lum'])
#     # Gets the start times of each stimulus.
#     st = np.append(
#         frameChanges[1::],
#         frameChanges[-1] + np.median(np.diff(frameChanges)),
#     )
#
#     et = frameChanges
#
#     # Checks if number of frames and stimuli match (if not, there
#     # could have been an issue with the photodiode, check if there
#     # are irregular frames in the photodiode trace).
#     if len(stimProps) != len(st):
#         # raise ValueError(
#         #     "Number of frames and stimuli do not match. Skpping"
#         # )
#         warnings.warn("Number of frames and stimuli do not match")
#         if (len(et) == len(stimProps)):
#             warnings.warn(
#                 "Assuming there was a false photodiode rise in the beginning, but check!")
#             st = frameChanges[0:-1:2].reshape(-1, 1).copy()
#             # Gets the end times  of each stimulus.
#             # et = frameChanges[2::2].reshape(-1, 1).copy()
#     # Adds the start and end times from above to the respective
#     # lists.
#
#
#
#     return {"Luminance.startTime.npy": st.reshape(-1, 1).copy(),
#             "Luminance.endTime.npy": et.reshape(-1, 1).copy(),
#             "Luminance.luminance.npy": stimProps.Lum.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "LuminanceExp.intervals.npy": np.atleast_2d([st[0], et[-1]]).T,
#             "LuminanceExp.description.npy": "Luminance",
#             }

# def stimulus_gratings_reward(directory, frameChanges):
#
#     # Gets the identity of the stimuli (see function for
#     # further details).
#     stimProps = get_stimulus_info(directory)
#     # Gets the start times of each stimulus.
#     st = frameChanges[::2].reshape(-1, 1).copy()
#     # Gets the end times  of each stimulus.
#     et = frameChanges[1::2].reshape(-1, 1).copy()
#
#     # Checks if number of frames and stimuli match (if not, there
#     # could have been an issue with the photodiode, check if there
#     # are irregular frames in the photodiode trace).
#     if len(stimProps) != len(st):
#         # raise ValueError(
#         #     "Number of frames and stimuli do not match. Skpping"
#         # )
#         warnings.warn("Number of frames and stimuli do not match")
#         if (len(et) == len(stimProps)):
#             warnings.warn(
#                 "Assuming there was a false photodiode rise in the beginning, but check!")
#             st = frameChanges[0:-1:2].reshape(-1, 1).copy()
#             # Gets the end times  of each stimulus.
#             # et = frameChanges[2::2].reshape(-1, 1).copy()
#     # Adds the start and end times from above to the respective
#     # lists.
#
#     if "Reward" in stimProps.columns:
#         reward = np.array(
#             [x in "True" for x in np.array(stimProps.Reward)]).reshape(-1, 1).astype(bool).copy()
#     else:
#         reward = np.ones_like(st) * np.nan
#
#     return {"gratings.startTime.npy": st,
#             "gratings.endTime.npy": et,
#             "gratings.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
#             "gratings.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.temporalF.npy": stimProps.TFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.contrast.npy": stimProps.Contrast.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.reward.npy": reward,
#             "gratingsExp.intervals.npy": [st[0], et[-1]],
#             "gratingsExp.description.npy": "Gratings",
#             }
#
#
# def stimulus_naturalImages(directory, frameChanges):
#     stimProps = get_stimulus_info(directory, 'FileIndex')
#
#     # Gets the start times of each stimulus.
#     # need to ignore the first stimulus since it's just the small square onset
#     st = frameChanges[1::2].reshape(-1, 1).copy()
#     # Gets the end times  of each stimulus.
#     et = frameChanges[2::2].reshape(-1, 1).copy()
#
#     if (len(st) > len(et)):
#         if (st[-1] > et[-1]):
#             st = st[:-1]
#
#     imgList = []
#     imgFile = glob.glob(os.path.join(directory, "imgList*.csv"))
#
#     if (len(imgFile) > 0):
#         imgList = pd.read_csv(imgFile[0], header=None).to_numpy()
#
#     return {"natural.startTime.npy": st,
#             "natural.endTime.npy": et,
#             "natural.fileIndex.npy": stimProps.FileIndex.to_numpy().reshape(-1, 1).astype(int).copy(),
#             "naturalExp.imgList.npy": {'imgList': np.array(imgList)},
#             "naturalExp.intervals.npy": [st[0, 0], et[-1, 0]]
#             }


# def stimulus_flicker(directory, frameChanges):
#     # Gets the identity of the stimuli (see function for
#     # further details).
#     # stimProps = get_stimulus_info(directory)
#     flicker_stimType = np.zeros(60)  # Asuming 10 reps per each contrast block
#
#     flicker_stimType[0::2] = 0.05  # SD from Low contrast
#     flicker_stimType[1::2] = 0.175  # SD from High contrast
#     flicker_stimType = flicker_stimType.reshape(-1, 1)
#
#     # Checks if number of frames and stimuli match (if not, there
#     # could have been an issue with the photodiode, check if there
#     # are irregular frames in the photodiode trace).
#
#     if (len(flicker_stimType)+1) == len(frameChanges):
#
#         # Gets the start times of each stimulus.
#         st = frameChanges[:-1].reshape(-1, 1).copy()
#         # Gets the end times  of each stimulus.
#         et = frameChanges[1:].reshape(-1, 1).copy()
#     else:
#         warnings.warn(f'''Number of frames and stimuli do not match.
#                       Assuming there was a false photodiode rise at the end, but check!''')
#
#         st = frameChanges[0:-2].reshape(-1, 1).copy()
#         et = frameChanges[1:-1].reshape(-1, 1).copy()
#
#     return {"flicker.startTime.npy": st,
#             "flicker.endTime.npy": et,
#             "flicker.contrast.npy": flicker_stimType,
#             "flickerExp.intervals.npy": [st[0], et[-1]],
#             }
#
#
# def stimulus_oddball(directory, frameChanges):
#     # Gets the identity of the stimuli (see function for
#     # further details).
#     stimProps = get_stimulus_info(directory)
#     # Gets the start times of each stimulus.
#     st = frameChanges[::2].reshape(-1, 1).copy()
#     # Gets the end times  of each stimulus.
#     et = frameChanges[1::2].reshape(-1, 1).copy()
#
#     # Checks if number of frames and stimuli match (if not, there
#     # could have been an issue with the photodiode, check if there
#     # are irregular frames in the photodiode trace).
#     if (len(stimProps)+1 == len(st)) & (len(stimProps) == len(et)):
#         # raise ValueError(
#         #     "Number of frames and stimuli do not match. Skpping"
#         # )
#         warnings.warn("Number of frames and stimuli do not match")
#         if (len(et) == len(stimProps)):
#             warnings.warn(
#                 "Assuming there was a false photodiode rise in the end, but check!")
#             st = frameChanges[0:-1:2].reshape(-1, 1).copy()
#             et = frameChanges[1:-1:2].reshape(-1, 1).copy()
#
#     elif (len(stimProps)+1 == len(st)) & (len(stimProps)+1 == len(et)):
#         warnings.warn("Number of frames and stimuli do not match")
#         warnings.warn(
#                 "Assuming there was 2 false photodiode rise: beggining and end, but check!")
#         st = frameChanges[1:-1:2].reshape(-1, 1).copy()
#         et = frameChanges[2:-1:2].reshape(-1, 1).copy()
#
#     if "Reward" in stimProps.columns:
#         reward = np.array(
#             [x in "True" for x in np.array(stimProps.Reward)]).reshape(-1, 1).astype(bool).copy()
#     else:
#         reward = np.ones_like(st) * np.nan
#
#     return {"gratings.startTime.npy": st,
#             "gratings.endTime.npy": et,
#             "gratings.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
#             "gratings.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.temporalF.npy": stimProps.TFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.contrast.npy": stimProps.Contrast.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratings.reward.npy": reward,
#             "gratingsExp.intervals.npy": np.atleast_2d([st[0], et[-1]]).T,
#             "gratingsExp.description.npy": 'Oddball'
#             }
#
#
# def stimulus_gratingsStep(directory, frameChanges):
#     '''
#     Step increasing/decreasing TFreq
#     '''
#     stimProps = get_stimulus_info(directory)
#     # Gets the start times of each stimulus.
#     st = frameChanges[::2].reshape(-1, 1).copy()
#     # Gets the end times  of each stimulus.
#     et = frameChanges[1::2].reshape(-1, 1).copy()
#
#     # Checks if number of frames and stimuli match (if not, there
#     # could have been an issue with the photodiode, check if there
#     # are irregular frames in the photodiode trace).
#     if len(stimProps) != len(st):
#
#         warnings.warn("Number of frames and stimuli do not match")
#         if (len(et) == len(stimProps)):
#             warnings.warn(
#                 "Assuming there was a false photodiode rise in the end, but check!")
#             st = frameChanges[0:-1:2].reshape(-1, 1).copy()
#
#     # Adds the start and end times from above to the respective
#     # lists.
#
#     return {"gratingsStep.startTime.npy": st,
#             "gratingsStep.endTime.npy": et,
#             "gratingsStep.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
#             "gratingsStep.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsStep.startTemporalF.npy": stimProps.TFreqStart.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsStep.endTemporalF.npy": stimProps.TFreqEnd.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsStep.contrast.npy": stimProps.Contrast.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsStepExp.intervals.npy": [st[0], et[-1]],
#             }
#
#
# def stimulus_gratingsContrastStep(directory, frameChanges):
#
#     stimProps = get_stimulus_info(directory)
#     # Gets the start times of each stimulus.
#     st = frameChanges[::2].reshape(-1, 1).copy()
#     # Gets the end times  of each stimulus.
#     et = frameChanges[1::2].reshape(-1, 1).copy()
#
#     # Checks if number of frames and stimuli match (if not, there
#     # could have been an issue with the photodiode, check if there
#     # are irregular frames in the photodiode trace).
#     if len(stimProps) != len(st):
#
#         warnings.warn("Number of frames and stimuli do not match")
#         if (len(et) == len(stimProps)):
#             warnings.warn(
#                 "Assuming there was a false photodiode rise in the end, but check!")
#             st = frameChanges[0:-1:2].reshape(-1, 1).copy()
#
#     # Adds the start and end times from above to the respective
#     # lists.
#
#     return {"gratingsContrastStep.startTime.npy": st,
#             "gratingsContrastStep.endTime.npy": et,
#             "gratingsContrastStep.direction.npy": stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy(),
#             "gratingsContrastStep.spatialF.npy": stimProps.SFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsContrastStep.temporalF.npy": stimProps.TFreq.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsContrastStep.contrastStart.npy": stimProps.ContrastStart.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsContrastStep.contrastEnd.npy": stimProps.ContrastEnd.to_numpy().reshape(-1, 1).astype(float).copy(),
#             "gratingsContrastStep.intervals.npy": [st[0], et[-1]],
#             }
#
#
# def stimulus_classificationExtended(directory, frameChanges):
#     retinal_et = np.append(
#         frameChanges[1::],
#         frameChanges[-1] + (frameChanges[14] - frameChanges[13]),
#     )
#
#     retinal_st = frameChanges
#
#     retinal_stimType = np.empty(
#         (len(frameChanges), 1), dtype=object
#     )
#     retinal_stimType[18::19] = "Off"
#     retinal_stimType[0::19] = "On"
#     retinal_stimType[1::19] = "Off"
#     retinal_stimType[2::19] = "Grey"
#     retinal_stimType[3::19] = "ChirpF"
#     retinal_stimType[4::19] = "Grey"
#     retinal_stimType[5::19] = "ChirpCIncreasing"
#     retinal_stimType[6::19] = "Grey"
#     retinal_stimType[7::19] = "ChirpCDecreasing"
#     retinal_stimType[8::19] = "Grey"
#     retinal_stimType[9::19] = "ChirpCIncreasing2Hz"
#     retinal_stimType[10::19] = "Grey"
#     retinal_stimType[11::19] = "ChirpCDecreasing2Hz"
#     retinal_stimType[12::19] = "Grey"
#     retinal_stimType[13::19] = "Off"
#     retinal_stimType[14::19] = "Blue"
#     retinal_stimType[15::19] = "Off"
#     retinal_stimType[16::19] = "Green"
#     retinal_stimType[17::19] = "Off"
#
#     return {'fullFieldExtended.startTime.npy': retinal_st.reshape(-1, 1).copy(),
#             "fullFieldExtended.endTime.npy": retinal_et.reshape(-1, 1).copy(),
#             "fullFieldExtended.stim.npy": retinal_stimType.copy(),
#             "fullFieldExtended.intervals.npy": [retinal_st[0], retinal_et[-1]]
#             }
    

stimulus_processing_dictionary = {
    "Gratings": stimulus_gratings,
    "Circles": stimulus_circles,
    "Retinal": stimulus_classification,
    "Sparse": stimulus_sparse,
    "Spont": stimulus_spont,
    "SpontGrey": stimulus_spont_grey,
    # "GratingsReward": stimulus_gratings_reward,
    # "NaturalImages": stimulus_naturalImages,
    # "Flicker": stimulus_flicker,
    # "Oddball": stimulus_oddball,
    # "GratingsStep": stimulus_gratingsStep,
    # "GratingsLuminance": stimulus_gratingsLuminance,
    # "Luminance": stimulus_Luminance,
    # "GratingsContrastStep": stimulus_gratingsContrastStep,
    # "RetinalExtended": stimulus_classificationExtended,
    
}
