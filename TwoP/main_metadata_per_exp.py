import glob
import sys
from contextlib import redirect_stdout, redirect_stderr
import cv2
import numpy as np
import yaml
import argparse
import os
import pandas as pd
from numpy import bool, ndarray
import matplotlib.pyplot as plt

from Bonsai_TwoP import extract_data
from TwoP import suite2p_compat, general


NIDAQ_SAMPLINGRATE = 1000
ARDUINO_SAMPLINGRATE = 1000


def determine_durations(protocol, times_up, times_down):
    if protocol == "gratings":
        starts = times_down
        n_pairs = min(len(times_down), len(times_up))
        duration = times_up[:n_pairs] - times_down[:n_pairs]
        isi = times_down[1:n_pairs] - times_up[:n_pairs-1]
    elif protocol == "circles":
        starts = np.sort(np.concatenate((times_up, times_down)), axis=0)
        duration = np.diff(starts, axis=0)
        duration = np.append(duration, np.nanmedian(duration))
        isi = []
    elif protocol == "fullField":
        t = np.sort(np.concatenate((times_up, times_down)), axis=0)
        end = (len(t) // 13) * 13
        t = t[:end]
        starts = t[::13]
        duration = t[12::13] - t[::13]
        isi = t[13::13] - t[12:-1:13]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return duration, isi, starts


def report_timing_mismatch(protocol, times_down, times_up, num_trials, outliers):
    """
    Print diagnostics when timing arrays and trial count do not match.
    """
    n_down = len(times_down)
    n_up = len(times_up)
    mismatch = False

    if protocol == "gratings":
        mismatch = (n_down != num_trials) or (n_down != n_up)
    elif protocol in ("circles", "fullField"):
        mismatch = (n_down + n_up) != num_trials

    # Only run diagnostics when mismatch exists.
    if mismatch:
        print("  Timing mismatch detected:")
        print(f"    num_trials: {num_trials}")
        print(f"    len(times_down): {n_down}")
        print(f"    len(times_up): {n_up}")
    else:
        print("  No timing mismatch detected.")

    if not outliers:  # outliers were manually set to False (or never existed)
        return mismatch, outliers


    duration, isi, starts = determine_durations(protocol, times_up, times_down)
    med_duration = np.nanmedian(duration)

    # Times where duration is 0.1 s shorter or longer than median.
    bad_duration = np.abs(duration - med_duration) > 0.1
    bad_duration_idx = np.where(bad_duration)[0]
    bad_duration_times = starts[bad_duration]
    if len(bad_duration_times) > 0:
        print(f"  Outliers: duration > 0.1 s from median {med_duration:.6f} s:")
        print(f"    Times: {bad_duration_times.tolist() if len(bad_duration_times) else []}")
        print(f"    Indices: {bad_duration_idx.tolist() if len(bad_duration_idx) else []}")
    else:
        outliers = False

    # ISI from consecutive trials: times_down[1:] - times_up[:-1]
    if len(isi) > 0:
        med_isi = np.nanmedian(isi)
        p10 = np.nanpercentile(isi, 10)
        p90 = np.nanpercentile(isi, 90)

        # 0.5 s shorter than p10 OR 0.5 s longer than p90.
        bad_isi = (isi < (p10 - 0.5)) | (isi > (p90 + 0.5))
        bad_isi_times = starts[1:][bad_isi]  # start times of following stimulus
        if len(bad_isi_times) > 0:
            print(f"  ISI < p10-0.5 s or > p90+0.5 s from {med_isi:.6f} s (next stimulus start times):")
            print(f"    {bad_isi_times.tolist() if len(bad_isi_times) else []}")
        else:
            outliers = False

    return mismatch, outliers


def review_timing_with_user(stimulus_name, num_trials, times_up, times_down, time_nidaq, photodiode):
    """
    Non-interactive-edit workflow:
    1) Show plot
    2) Pause so user can manually edit vectors in debugger/console
    3) Run mismatch report
    4) Ask for confirmation or another edit cycle
    """

    # Ensure arrays
    times_down = np.asarray(times_down).copy()
    times_up = np.asarray(times_up).copy()
    bad_trials = []
    outliers = True

    while True:
        # Run your diagnostics on current vectors
        mismatch, outliers = report_timing_mismatch(stimulus_name, times_down, times_up, num_trials, outliers)
        if not mismatch and not outliers:
            break

        # Plot current state
        plt.figure(figsize=(12, 4))
        # try:
        #     manager = plt.get_current_fig_manager()
        #     manager.window.wm_geometry("+2000+100")
        # except Exception:
        #     try:
        #         manager.window.setGeometry(2200, 100, 1200, 500)
        #     except Exception:
        #         pass
        plt.plot(time_nidaq, photodiode, color="0.2", lw=1, label="photodiode")
        if len(times_down):
            plt.plot(times_down, np.full(len(times_down), np.nanpercentile(photodiode, 70)), "rv", label="times_down")
        if len(times_up):
            plt.plot(times_up, np.full(len(times_up), np.nanpercentile(photodiode, 30)), "g^", label="times_up")
        plt.title(f"{stimulus_name}: review times_down/times_up")
        plt.xlabel("time (s)")
        plt.ylabel("photodiode")
        plt.legend(loc="best")
        plt.tight_layout()
        # plt.show(block=False)

        print("\n--- Timing review ---")
        print("Entering debugger. Edit `times_down` and/or `times_up` as needed.")
        print("times_down = np.delete(times_down, [])   # remove by indices")
        print("times_up = np.delete(times_up, [])   # remove by indices")
        print("outliers = False")
        print("bad_trials = []")

        plt.close("all")  # ALWAYS SET BREAKPOINT HERE!

    return times_up, times_down, bad_trials


def correct_times_stimuli(stimulus_name, times_up, times_down):
    if stimulus_name == "gratings":
        # Gratings start with black squares (times_down) and end with white square (times_up).
        # If number of starts and ends doesn't match, ignore the excess.
        if times_up[0] < times_down[0]:  # if photodiode goes up first, ignore those
            times_up = times_up[times_up[:, 0] > times_down[0], 0]
        if times_down[-1] > times_up[-1]:  # if photodiode goes down at the end, ignore those
            times_down = times_down[times_down[:, 0] < times_up[-1]]
    # elif stimulus_name == "circles": # New circle starts with every switch -> no rules.

    return times_up, times_down


def find_violation_trials(protocol, photodiode_up: np.ndarray, photodiode_down: np.ndarray,
                          violation_up: np.ndarray, violation_down: np.ndarray) -> np.ndarray:
    """
    For each violation time in violation_up, find the index of the first entry
    in photodiode_up that is strictly greater than that violation time.

    Parameters
    ----------
    photodiode_up : np.ndarray
        Sorted array of upward crossing times (s).
    violation_up : np.ndarray
        Array of violation times (s) — each is a time at which a spurious
        upward crossing was detected.

    Returns
    -------
    np.ndarray of int
        Indices into photodiode_up of the crossing that immediately follows
        each violation time. Violations with no following entry are omitted.
    """
    indices = []
    if protocol == "gratings":
        for vt in violation_up:
            candidates = np.where(photodiode_up > vt)[0]
            if len(candidates) > 0:
                indices.append(candidates[0])
    elif protocol in ("circles", "fullField"):
        for vt in violation_up:
            candidates = np.where(photodiode_down < vt)[0]
            if len(candidates) > 0:
                indices.append(candidates[-1])  # last photodiode_down before violation time
        for vt in violation_down:
            candidates = np.where(photodiode_up < vt)[0]
            if len(candidates) > 0:
                indices.append(candidates[-1])  # last photodiode_up before violation time
        indices.sort()

    return indices


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
        })


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
    data_paths = db["data_path"]

    # Determine number of frames of each experiment.
    num_frames_exp = db["frames_per_folder"]
    # Gets how many planes were imaged.
    num_planes = db["nplanes"]

    for ind_exp, f_exp in enumerate(data_paths):
        if ind_exp < 0:
            continue
        exp = os.path.split(f_exp)[-1]
        print(f"  Experiment: {exp}")

        exp_folder = os.path.join(bonsai_folder, exp)
        if not os.path.isdir(exp_folder):
            print(f"    WARNING: {exp_folder} not found. Skipping whole dataset")
            return

        exp_output_folder = os.path.join(output_folder, exp)
        os.makedirs(exp_output_folder, exist_ok=True)

        # Load NiDAQ data.
        nidaq, chans_nidaq, time_nidaq = extract_data.get_nidaq_channels(exp_folder, sampling_rate=NIDAQ_SAMPLINGRATE)
        np.save(os.path.join(exp_output_folder, "time_nidaq.npy"), time_nidaq)

        # Load Arduino data.
        arduino, chans_arduino, time_arduino_unsynced = extract_data.get_arduino_data(exp_folder, sampling_rate=ARDUINO_SAMPLINGRATE)

        # Sync Arduino to NiDAQ time.
        nidaq_sync = nidaq[:, chans_nidaq == "sync"][:, 0]
        arduino_sync = arduino[:, chans_arduino == "sync"][:, 0]
        time_arduino = extract_data.arduino_delay_compensation(nidaq_sync, arduino_sync, time_nidaq, time_arduino_unsynced)
        np.save(os.path.join(exp_output_folder, "time_arduino.npy"), time_arduino)

        # Gets stimulus information.
        # Types: Gratings, Circles, Retinal, Spont
        properties_file = glob.glob(os.path.join(exp_folder, "props*.csv"))
        property_names = np.loadtxt(properties_file[0], dtype=str, delimiter=",", ndmin=2).T[0]
        stimuli, num_trials, protocol, times_stim_bonsai = extract_data.process_stimulus(property_names[0], exp_folder)
        if times_stim_bonsai is not None:
            np.save(os.path.join(exp_output_folder, f"{protocol}.timestamps_bonsai.npy"),
                    times_stim_bonsai / NIDAQ_SAMPLINGRATE)

        # Stimulus times.
        if not "Screen" in protocol:
            bad_trials, photodiode_down, photodiode_up = extract_stimulus_times(
                nidaq[:, chans_nidaq == "photodiode"], time_nidaq, num_trials, protocol,
                os.path.join(output_folder, exp, "plots"))

            # Add photodiode-based timing results directly to stimulusResults
            if protocol == "gratings":
                times_stim = photodiode_down
                stimuli[f"{protocol}.intervals.npy"] = np.column_stack((photodiode_down, photodiode_up))
            elif protocol in ("circles", "fullField"):
                times_stim = np.sort(np.concatenate((photodiode_down, photodiode_up)), axis=0)
                stimuli[f"{protocol}.times.npy"] = times_stim

            stimuli[f"{protocol}.badTrials.npy"] = bad_trials

        stimuli[f"recording.{protocol}_intervals.npy"] = np.atleast_2d([0, time_nidaq[-1,0]]).T

        # Save stimulus-related information
        for key, value in stimuli.items():
            save_path = os.path.join(exp_output_folder, key)
            np.save(save_path, np.asarray(value))

        if protocol == "fullField":
            cycle = np.array(
                ["On", "Off", "Grey", "ChirpF", "Grey", "ChirpC", "Grey",
                 "Off", "Blue", "Off", "Green", "Off", "Off"],
                dtype=object,
            )
            n = len(times_stim)
            stim_names = np.resize(cycle, n).reshape(-1, 1)  # same shape as before: (n, 1)

            # Write one label per line (single-column CSV).
            csv_path = os.path.join(exp_output_folder, "fullField.stimNames.csv")
            np.savetxt(csv_path, stim_names.ravel(), fmt="%s", delimiter=",")

        # Get number of video frames
        vfile = glob.glob(os.path.join(f_exp, "Video[0-9]*.avi"))[0]  # eye
        video_eye = cv2.VideoCapture(vfile)
        vfile = glob.glob(os.path.join(f_exp, "Video[a-zA-Z]*.avi"))[0]  # body
        video_body = cv2.VideoCapture(vfile)
        # number of frames
        nframes_eye = int(video_eye.get(cv2.CAP_PROP_FRAME_COUNT))
        nframes_body = int(video_body.get(cv2.CAP_PROP_FRAME_COUNT))

        # Arduino data: extract video frame times.
        camera_eye = arduino[:, chans_arduino == "camera1"][:, 0]
        camera_body = arduino[:, chans_arduino == "camera2"][:, 0]
        times_eye = extract_data.assign_frame_time(camera_eye, time_arduino, th=0.5, plot=False)
        times_body = extract_data.assign_frame_time(camera_body, time_arduino, th=0.5, plot=False)

        # Bonsai log: extract video frame times.
        event_names = ["Video$", "Video,[0-9]*", "Analog*"]
        if "Spont" in property_names[0]:
            event_names = ["VideoFrame", "Video,[0-9]*", "NiDaq*"]
        event_times = extract_data.get_recorded_video_times(
            exp_folder, event_names, ["EyeVid", "BodyVid", "NI"])
        times_eye_bonsai = event_times["EyeVid"].astype(float).reshape(-1,1) / NIDAQ_SAMPLINGRATE
        times_body_bonsai = event_times["BodyVid"].astype(float).reshape(-1,1) / NIDAQ_SAMPLINGRATE

        np.save(os.path.join(exp_output_folder, "eye.timestamps.npy"), times_eye)
        np.save(os.path.join(exp_output_folder, "eye.timestamps_bonsai.npy"), times_eye_bonsai)
        np.save(os.path.join(exp_output_folder, "eye.nframes.npy"), nframes_eye)

        np.save(os.path.join(exp_output_folder, "body.timestamps.npy"), times_body)
        np.save(os.path.join(exp_output_folder, "body.timestamps_bonsai.npy"), times_body_bonsai)
        np.save(os.path.join(exp_output_folder, "body.nframes.npy"), nframes_body)

        np.save(os.path.join(exp_output_folder, "bonsai.ntimestamps.npy"), len(event_times["NI"]))

        # Times of imaged frames.
        time_plane, plane_delays = extract_frametimes(
            nidaq[:, chans_nidaq == "frameclock"], time_nidaq, num_frames_exp[ind_exp], num_planes)
        np.save(os.path.join(exp_output_folder, "2pCalcium.timestamps.npy"), time_plane.reshape(-1, 1))
        np.save(os.path.join(exp_output_folder, "2pPlanes.delay.npy"), plane_delays.reshape(-1, 1))

        # Lick spout
        if "lick" in chans_nidaq:
            np.save(os.path.join(exp_output_folder, "spout.licks.npy"), nidaq[:, chans_nidaq == "lick"].reshape(-1, 1))

        # Wheel movement.
        movement1 = arduino[:, chans_arduino == "rotary1"][:, 0]
        movement2 = arduino[:, chans_arduino == "rotary2"][:, 0]
        velocity = extract_data.detect_wheel_move(movement1, movement2, time_arduino)
        np.save(os.path.join(exp_output_folder, "wheel.velocity.npy"), velocity.reshape(-1, 1))


def extract_stimulus_times(signal, time_nidaq, num_trials, protocol, plot_folder):
    # (1) Switches of photodiode on monitor to white and black. Violations occur if two consecutive switches occur
    #     in the same direction (first switch didn't reach top or bottom values).
    photodiode_up, photodiode_down, violation_up, violation_down = extract_data.detect_photodiode_changes(
        signal, time_nidaq, plot_folder=plot_folder)

    # (2) Apply your automatic corrections.
    photodiode_up, photodiode_down = correct_times_stimuli(protocol, photodiode_up, photodiode_down)

    # (3) Let user review data if problematic.
    # NOTE: SET BREAKPOINT IN THIS FUNCTION!
    photodiode_up, photodiode_down, invalid_trials = review_timing_with_user(
        protocol, num_trials, photodiode_up, photodiode_down, time_nidaq,
        signal[:, 0])

    # (4) Identify trials where violations (short switches of photodiode) occur.
    violation_trials = find_violation_trials(protocol, photodiode_up, photodiode_down, violation_up, violation_down)

    bad_trials = np.zeros((num_trials, 1), dtype=bool)
    bad_trials[np.array(sorted(set(invalid_trials + violation_trials)), dtype=int)] = True
    return bad_trials, photodiode_down, photodiode_up


def extract_frametimes(frameclock: ndarray, time: ndarray, num_frames: int, num_planes: int):
    # Extract time point of each imaged frame.
    time_frames = extract_data.assign_frame_time(frameclock, time)

    # Determine delays of all planes relative to first plane.
    frame_dur = np.median(np.diff(time_frames))
    plane_delays = np.arange(num_planes) * frame_dur

    # Only consider times of first plane. Check whether number of time points matches number of imaged frames.
    time_plane = time_frames[::num_planes]
    if len(time_plane) - num_frames == 1:  # first plane is one frame longer than last plane -> ignore last frame
        time_plane = time_plane[:-1]
    elif len(time_plane) != num_frames:
        print(f"    WARNING: Number of frame times ({len(time_plane)}) does not match " +
              f"number of imaged frames ({num_frames}). Ignore excess times, or add extra.")
        if len(time_plane) > num_frames:
            time_plane = time_plane[:num_frames]
        else:
            extra_times = (np.arange(num_frames - len(time_plane)) + 1) * np.median(np.diff(time_plane)) + time_plane[-1]
            time_plane = np.concatenate([time_plane, extra_times])

    return time_plane, plane_delays


def preprocess(config: dict, datasets: pd.DataFrame):
    # Choose a stable directory for all run logs.
    # If output template supports placeholders only, this creates .../ALL/ALL/
    log_dir = config["directories"]["output"].format(Name="ALL", Date="ALL")
    # Alternative fallback:
    # log_dir = os.path.join(os.getcwd(), "logs")

    session_log_path = general.make_incremental_log_path(log_dir, base_name="main_metadata", ext=".txt")
    log_file, tee = general.make_logger(session_log_path)
    try:
        with redirect_stdout(tee), redirect_stderr(tee):
            for i in range(len(datasets)):
                if not datasets.loc[i]["Process"]:
                    continue

                print(f"Processing {datasets.loc[i]['Name']} {datasets.loc[i]['Date']}...")
                paths = general.make_paths(datasets.loc[i], config["directories"])
                plane_folder = glob.glob(os.path.join(paths["suite2p"], "plane*"))[-1]
                db = suite2p_compat.load_parameters(plane_folder)[0]
                process_metadata_directory(paths["raw"], paths["output"], db)
    finally:
        log_file.close()


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

# NOTE: ALWAYS PUT BREAKPOINT IN FUNCTION review_timing_with_user
