import glob
import os
import traceback
import warnings
from typing import Any, Union

import cv2
import numpy
import numpy as np
import pandas

import yaml
import argparse
import os
import pandas as pd
from numpy import bool, complexfloating, dtype, floating, ndarray, number, signedinteger, timedelta64, unsignedinteger

from Bonsai.extract_data import get_piezo_data, get_nidaq_channels, assign_frame_time, detect_photodiode_changes, \
    process_stimulus, get_arduino_data, arduino_delay_compensation, detect_wheel_move, get_recorded_video_times, \
    save_stimuli
from TwoP import suite2p_compat
from user_defs import create_2p_processing_ops


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset_list(path):
    return pd.read_csv(
        path,
        dtype={
            "Name": str,
            "Date": str,
            "Process": bool
        }
    )


def make_paths(dataset: pd.Series, directories: dict) -> dict:
    """
    Build paths for a single experiment row.

    Parameters
    ----------
    dataset : pandas Series
        A single row (experiment) from the preprocess DataFrame.
    directories : dict
        Base directories dictionary.

    Returns
    -------
    dict
        Paths for tiffs, suite2p, piezo, and output.
    """
    subject = str(dataset.Name)
    date = str(dataset.Date)
    paths = {
        "raw": directories["raw"].format(Name=subject, Date=date),
        "suite2p": directories["suite2p"].format(Name=subject, Date=date),
        "output": directories["output"].format(Name=subject, Date=date),
    }

    if not os.path.exists(paths["suite2p"]):
        paths["suite2p"] = None
        print(
            f"NOTE: Suite2p directory for {dataset['Name']} {dataset['Date']} not found. Skipping.\n\n"
        )
        return paths

    os.makedirs(paths["output"], exist_ok=True)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess TwoP meta-data')
    parser.add_argument('--config', type=str,
                        required=False, default='preprocess.yaml',
                        help='Path to preprocess.yaml')
    return parser.parse_args()


def process_metadata_directory(bonsai_folder: str, output_folder: str, db: dict):
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
    bonsai_folder : str
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
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Analyzed experiments
    exp_folders = db["data_path"]

    # Determine number of frames of each experiment.
    num_frames_exp = db["frames_per_folder"]
    # Gets how many planes were imaged.
    num_planes = db["nplanes"]

    # Lists of outputs.
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

    lastFrame = 0  # initialize frame count
    for ind_exp, f_exp in enumerate(exp_folders):
        exp = os.path.split(f_exp)[-1]
        print(f"  Experiment: {exp}")

        folder = os.path.join(bonsai_folder, exp)
        if not os.path.isdir(folder):
            print(f"    WARNING: {folder} not found. Skipping whole dataset")
            return

        nidaq, chans, nt = get_nidaq_channels(f_exp, plot=True)

        # Times of imaged frames.
        frameclock, imagedFrames, planeTimeDelta = extract_frametimes(chans, ind_exp, nidaq, num_frames_exp,
                                                                          num_planes)
        frameTimes.append(imagedFrames + lastFrame)

        # Stimulus on- and offset times.
        frameChanges = detect_photodiode_changes(nidaq[:, chans == "photodiode"], plot=True)
        frameChanges += lastFrame


        # Gets stimulus information.
        sparseFile = glob.glob(os.path.join(f_exp, "SparseNoise*"))
        propsFile = glob.glob(os.path.join(f_exp, "props*.csv"))
        propTitles = np.loadtxt(
            propsFile[0], dtype=str, delimiter=",", ndmin=2
        ).T
        if ("Spont" in propTitles[0]) | (len(sparseFile) != 0):
            sparseNoise = True
        else:
            sparseNoise = False

        stimulusResults = process_stimulus(propTitles, f_exp, frameChanges)
        stimulusProps.append(stimulusResults)
        stimulusTypes.append(propTitles[0][0])

        # lick spout
        lickSpout = np.ones_like(frameclock) * np.nan
        if ("lick" in chans):
            lickSpout = nidaq[:, chans == "lick"]
        licks.append(lickSpout)
        lickTimes.append(nt + lastFrame)

        # Count number of frames in eye and body videos
        nframes1 = np.nan
        nframes2 = np.nan
        # Get actual video data
        vfile = glob.glob(os.path.join(
            f_exp, "Video[0-9]*.avi"))  # eye
        if (len(vfile) > 0):
            vfile = vfile[0]
            video1 = cv2.VideoCapture(vfile)
            nframes1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
        vfile = glob.glob(os.path.join(
            f_exp, "Video[a-zA-Z]*.avi"))  # body
        if (len(vfile) > 0):
            vfile = vfile[0]
            video2 = cv2.VideoCapture(vfile)
            nframes2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

        # Arduino data: align to niDaq.
        # Gets the arduino data (see function for details).
        ardData, ardChans, at = get_arduino_data(f_exp)
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

        # Arduino data: extract wheel movement.
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

        # Arduino data: extract video frame times.
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
            f_exp,
            logColNames,
            ["EyeVid", "BodyVid", "NI"],
        )
        cam1Frames = colNiTimes["EyeVid"].astype(float) / 1000
        cam2Frames = colNiTimes["BodyVid"].astype(float) / 1000
        # Get actual video data
        vfile = glob.glob(os.path.join(
            f_exp, "Video[0-9]*.avi"))[0]  # eye
        video1 = cv2.VideoCapture(vfile)
        vfile = glob.glob(os.path.join(
            f_exp, "Video[a-zA-Z]*.avi"))[0]  # body
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

        # Count total time across experiments.
        lastFrame = nt[-1] + lastFrame

    np.save(os.path.join(output_folder, "calcium.timestamps.npy"), np.hstack(frameTimes).reshape(-1, 1))
    np.save(os.path.join(output_folder, "planes.delay.npy"), planeTimeDelta.reshape(-1, 1))

    # concatante stimuli and save them
    save_stimuli(output_folder, stimulusTypes, stimulusProps)

    if len(wheelTimes) > 0:
        np.save(os.path.join(output_folder, "wheel.timestamps.npy"), np.hstack(wheelTimes).reshape(-1, 1))
        np.save(os.path.join(output_folder, "wheel.velocity.npy"), np.hstack(velocity).reshape(-1, 1))
    if len(faceTimes) > 0:
        np.save(os.path.join(output_folder, "eye.timestamps.npy"), np.hstack(faceTimes).reshape(-1, 1))
    if len(bodyTimes) > 0:
        np.save(os.path.join(output_folder, "body.timestamps.npy"), np.hstack(bodyTimes).reshape(-1, 1))
    if len(licks) > 0:
        np.save(os.path.join(output_folder, "spout.timestamps.npy"), np.hstack(lickTimes).reshape(-1, 1))
        np.save(os.path.join(output_folder, "spout.licks.npy"), np.vstack(licks).reshape(-1, 1))


def extract_frametimes(chans, ind_exp: int, nidaq, num_frames_exp, num_planes) -> tuple[Union[Union[
    ndarray[Any, dtype[Any]], ndarray[Any, dtype[bool]], ndarray[Any, dtype[unsignedinteger[Any]]], ndarray[
        Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[floating[Any]]], ndarray[
        Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[number[Any]]], ndarray[
        Any, dtype[timedelta64]], float], Any], Any, Union[Union[
    ndarray[Any, dtype[Any]], ndarray[Any, dtype[bool]], ndarray[Any, dtype[unsignedinteger[Any]]], ndarray[
        Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[floating[Any]]], ndarray[
        Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[number[Any]]], ndarray[
        Any, dtype[timedelta64]]], Any]]:
    # Gets the frame clock data.
    frameclock = nidaq[:, chans == "frameclock"]
    # Assigns a time in ms to a frame time (see function for details).
    frames = assign_frame_time(frameclock, plot=pops["plot"])
    # TODO: run the 5 lines below in debug mode.
    frameDiffMedian = np.median(np.diff(frames))
    # Take only first frames of each go.
    firstFrames = frames[::num_planes]
    imagedFrames = np.zeros(num_frames_exp[ind_exp]) * np.nan
    imagedFrames[: len(firstFrames)] = firstFrames
    planeTimeDelta = np.arange(num_planes) * frameDiffMedian
    return frameclock, imagedFrames, planeTimeDelta


def preprocess(config: dict, datasets: pd.DataFrame):
    for i in range(len(datasets)):
        if not datasets.loc[i]["Process"]:
            continue

        print(f"Processing {datasets.loc[i]['Name']} {datasets.loc[i]['Date']}...")
        paths = make_paths(datasets.loc[i], config["directories"])

        # Load parameters that were used to process data with Suite2p (from first plane in folder).
        db, settings = suite2p_compat.load_parameters_recording(paths["suite2p"])

        process_metadata_directory(paths["raw"], paths["output"], db)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        exit(1)
    conf = load_config(args.config)
    if not os.path.exists(conf["datasets"]):
        print(f"Dataset file {conf['datasets']} does not exist.")
        exit(1)
    ds = load_dataset_list(conf['datasets'])
    if ds is None:
        exit(1)
    preprocess(conf, ds)


