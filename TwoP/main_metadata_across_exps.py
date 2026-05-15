import ast
import glob
from contextlib import redirect_stdout, redirect_stderr
import yaml
import argparse
import os
import pandas as pd
import numpy as np

from TwoP import general


GAP = 5  # gap between experiments in seconds
BIN_ARDUINO = 0.01 # new bin size for arduino time data

file_names = {
    "time": ["time_nidaq.npy", "time_arduino.npy", "bonsai.ntimestamps.npy"],
    "traces": ["2pCalcium.timestamps.npy", "2pPlanes.delay.npy"],
    "eye": ["eye.timestamps.npy", "eye.timestamps_bonsai.npy", "eye.nframes.npy"],
    "body": ["body.timestamps.npy", "body.timestamps_bonsai.npy", "body.nframes.npy"],
    "wheel": ["wheel.velocity.npy"],
}
protocols = ["gratings", "circles", "fullField", "darkScreen", "grayScreen", "sparse", "luminance"]


def combine_metadata(path, exp_atropine):
    # Collect all data per experiment
    exp_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    rec_durations = []
    stimuli = []
    data = {
        "time": {},
        "traces": {},
        "eye": {},
        "body": {},
        "wheel": {}
    }
    # Initialize lists for each file
    for key in data.keys():
        for file_name in file_names[key]:
            data[key][file_name] = []

    for exp_folder in exp_folders:
        print(f"  Experiment: {exp_folder}")
        for key in data.keys():
            for file_name in file_names[key]:
                file_path = os.path.join(path, exp_folder, file_name)
                if os.path.exists(file_path):
                    data[key][file_name].append(np.load(file_path))
                else:
                    data[key][file_name].append(None)
                    print(f"    File {file_name} not found in {exp_folder}.")

        exp_num = int(''.join(filter(str.isdigit, exp_folder)))
        is_atropine = exp_num in exp_atropine

        stim_data = collect_stim_data(path, exp_folder, is_atropine, rec_durations)

        stimuli.append(stim_data)

    # Check whether eye and body timestamps may be swapped (compare number of video frames to TTL pulses)
    nt_eye = data["eye"]["eye.timestamps.npy"]
    nt_body = data["body"]["body.timestamps.npy"]
    check_videos_swapped(data, data["body"]["body.nframes.npy"], data["eye"]["eye.nframes.npy"], nt_body, nt_eye)

    # Try to sync DAQ time with bonsai time
    regression_bonsai = sync_bonsai_to_daq(data, stimuli)

    # For each experiment: check video time data (eye + body) and decide which times to use if any
    times_eye, source_eye = select_video_times("eye", data["eye"], data["time"], len(exp_folders),
                                               rec_durations, regression_bonsai)
    if not all(np.isnan(times_eye)):
        np.save(os.path.join(path, "eye.timestamps.npy"), times_eye)
        np.save(os.path.join(path, "eye.time_source.npy"), source_eye)

    times_body, source_body = select_video_times("body", data["body"], data["time"], len(exp_folders),
                                                 rec_durations, regression_bonsai)
    if not all(np.isnan(times_body)):
        np.save(os.path.join(path, "body.timestamps.npy"), times_body)
        np.save(os.path.join(path, "body.time_source.npy"), source_body)

    # Combine trace times, arduino time, and wheel data
    time_arduino, times_calcium, wheel_velocity = combine_daq_data(data, exp_folders, rec_durations)

    np.save(os.path.join(path, "2pCalcium.timestamps.npy"), times_calcium)
    np.save(os.path.join(path, "2pPlanes.delay.npy"),
            data["traces"]["2pPlanes.delay.npy"][0])  # same for all experiments
    if len(time_arduino) > 0:
        np.save(os.path.join(path, "wheel.timestamps.npy"), time_arduino)
        np.save(os.path.join(path, "wheel.velocity.npy"), wheel_velocity)

    # Combine stim info of several experiments if from same protocol
    combine_stimuli(path, rec_durations, stimuli)


def combine_stimuli(path, rec_durations, stimuli):
    # Loop over protocols
    for protocol in protocols:
        idx_exp = [i for i in range(len(stimuli)) if stimuli[i]["protocol"] == protocol]
        if not idx_exp:
            continue

        # Collect all keys across all experiments with this protocol
        all_keys = set()
        for i in idx_exp:
            all_keys.update(stimuli[i].keys())

        # Check which experiments are missing which keys
        for i in idx_exp:
            missing_keys = all_keys - set(stimuli[i].keys())
            if missing_keys:
                print(f"    Experiment {i}: missing keys {missing_keys}")

        all_keys.discard(f"{protocol}.timestamps_bonsai.npy")
        all_keys.discard("protocol")
        for key in all_keys:
            if "intervals.npy" in key:
                intv = np.zeros((0, 2))
                for i in idx_exp:
                    exp_start = 0 if i == 0 else sum(rec_durations[:i]) + i * GAP
                    intv = np.vstack((intv, stimuli[i][key] + exp_start))
                np.save(os.path.join(path, key), intv)
            elif "times.npy" in key:
                t = np.zeros((0, 1))
                for i in idx_exp:
                    exp_start = 0 if i == 0 else sum(rec_durations[:i]) + i * GAP
                    t = np.vstack((t, stimuli[i][key] + exp_start))
                np.save(os.path.join(path, key), t)
            elif "stimNames.csv" in key:
                names = []
                for i in idx_exp:
                    names.extend(stimuli[i][key])
                np.savetxt(os.path.join(path, key), names, fmt="%s", delimiter=",")
            else:
                if all(key in stimuli[i].keys() for i in idx_exp):
                    trial_data = np.vstack([stimuli[i][key] for i in idx_exp])
                    if key.endswith(".csv"):
                        np.savetxt(os.path.join(path, key), trial_data, fmt="%s", delimiter=",")
                    else:
                        np.save(os.path.join(path, key), trial_data)
                elif key == f"{protocol}Exp.description.npy":  # description was introduced for atropine experiments
                    trial_data = []
                    for i in idx_exp:
                        d = stimuli[i].get(key)
                        if d is None:
                            trial_data.append("Control")
                        else:
                            trial_data.append(d)

                    np.save(os.path.join(path, key), np.array(trial_data).reshape(-1, 1))

        rec_intervals = np.zeros((0, 2))
        for i in idx_exp:
            exp_start = 0 if i == 0 else sum(rec_durations[:i]) + i * GAP
            intv = np.array([0, float(rec_durations[i])]) + exp_start
            rec_intervals = np.vstack((rec_intervals, intv))
        np.save(os.path.join(path, f"recording.{protocol}_intervals.npy"), rec_intervals)


def combine_daq_data(data, exp_folders, rec_durations):
    times_calcium = np.zeros((0, 1))
    time_arduino = np.zeros((0, 1))
    wheel_velocity = np.zeros((0, 1))
    for i in range(len(exp_folders)):
        exp_start = 0 if i == 0 else sum(rec_durations[:i]) + i * GAP

        times_calcium = np.vstack((times_calcium, data["traces"]["2pCalcium.timestamps.npy"][i] + exp_start))

        t = data["time"]["time_arduino.npy"][i]
        if t is None or len(t) == 0:
            continue
        else:
            t = t  + exp_start
            time_binned = np.arange(np.ceil(t[0, 0] / BIN_ARDUINO) * BIN_ARDUINO, t[-1, 0], BIN_ARDUINO).reshape(-1, 1)
            time_arduino = np.vstack((time_arduino, time_binned))

            # Interpolate wheel velocity to binned time series
            vel = np.asarray(data["wheel"]["wheel.velocity.npy"][i], dtype=float).flatten()
            vel_interp = np.interp(time_binned.flatten(), t.flatten(), vel).reshape(-1, 1)
            wheel_velocity = np.vstack((wheel_velocity, vel_interp))
    return time_arduino, times_calcium, wheel_velocity


def sync_bonsai_to_daq(data, stimuli):
    regression_bonsai = {"stimuli": [], "eye": [], "body": []}
    # Go through stimuli
    for i in range(len(stimuli)):
        protocol = stimuli[i]["protocol"]
        if protocol in ("darkScreen", "greyScreen", "fullField"):
            regression_bonsai["stimuli"].append(None)
            continue

        times_photo = stimuli[i].get(f"{protocol}.times.npy")
        if times_photo is None:
            times_photo = stimuli[i].get(f"{protocol}.intervals.npy")
        if times_photo is not None:
            times_photo = times_photo[:, 0]
        else:
            print(f"  Exp {i} - {protocol}: no photodiode times found.")
            regression_bonsai["stimuli"].append(None)
            continue

        times_bonsai = stimuli[i].get(f"{protocol}.timestamps_bonsai.npy")
        if times_bonsai is None:
            print(f"  Exp {i} - {protocol}: no bonsai timestamps found.")
            regression_bonsai["stimuli"].append(None)
            continue

        if len(times_bonsai) != len(times_photo):
            print(
                f"  Exp {i} - {protocol}: #bonsai times ({len(times_bonsai)}) != #photodiode times ({len(times_photo)}).")
            regression_bonsai["stimuli"].append(None)
            continue

        # Fit linear regression between bonsai and photodiode times
        A = np.hstack([times_bonsai.reshape(-1, 1), np.ones((len(times_bonsai), 1))])
        result = np.linalg.lstsq(A, times_photo, rcond=None)
        m, c = result[0]  # m: slope, c: intercept
        mse = result[1][0] / len(times_bonsai) if len(result[1]) > 0 else np.nan
        regression_bonsai["stimuli"].append((m, c, mse))

    # Go through videos: just compare dt and offset of first frame
    for video in ("eye", "body"):
        for i in range(len(stimuli)):
            times_TTL = data[video][f"{video}.timestamps.npy"][i]
            if times_TTL is None or len(times_TTL) == 0:
                print(f"  Exp {i}: {video} video: no TTL times found.")
                regression_bonsai[f"{video}"].append(None)
                continue

            times_bonsai = data[video][f"{video}.timestamps_bonsai.npy"][i]
            if times_bonsai is None:
                print(f"  Exp {i}: {video} video: no bonsai timestamps found.")
                regression_bonsai[f"{video}"].append(None)
                continue

            dt_TTL = np.median(np.diff(times_TTL, axis=0))
            dt_bonsai = np.median(np.diff(times_bonsai, axis=0))

            time_arduino = data["time"]["time_arduino.npy"][i]
            times_diff = len(times_TTL) - len(times_bonsai)
            if times_diff < 0:  # more bonsai timestamps than TTL pulses
                # check whether early or late TTL pulses may be lost (because they continue beyond recording)
                if times_TTL[0] - dt_TTL > time_arduino[0]:
                    timepoint_TTL = times_TTL[0, 0]
                    timepoint_bonsai = times_bonsai[0, 0]
                elif times_TTL[-1] + dt_TTL < time_arduino[-1]:
                    timepoint_TTL = times_TTL[-1, 0]
                    timepoint_bonsai = times_bonsai[-1, 0]
                else:
                    print(
                        f"  Exp {i}: {video} video: #TTL times ({len(times_TTL)}) and #bonsai times ({len(times_bonsai)}) "
                        f"cannot be matched.")
                    regression_bonsai[f"{video}"].append(None)
                    continue
            elif times_diff > 0:
                print(f"  Exp {i}: {video} video: #TTL times ({len(times_TTL)}) > #bonsai times ({len(times_bonsai)}).")
                regression_bonsai[f"{video}"].append(None)
                continue
            else:
                timepoint_TTL = times_TTL[0, 0]
                timepoint_bonsai = times_bonsai[0, 0]

            # Estimate linear regression between bonsai and TTL times
            m = dt_TTL / dt_bonsai
            c = timepoint_TTL - timepoint_bonsai * m
            regression_bonsai[f"{video}"].append((m, c, 0.0))

    m = [regression_bonsai["eye"][i][0] for i in range(len(regression_bonsai["eye"]))
         if regression_bonsai["eye"][i] is not None]
    c = [regression_bonsai["eye"][i][1] for i in range(len(regression_bonsai["eye"]))
         if regression_bonsai["eye"][i] is not None]
    if len(m) > 0 or len(c) > 0:
        print(f"  For eye videos: bonsai factors: {', '.join(str(entry) for entry in m)}; "
              f"bonsai offsets: {', '.join(str(entry) for entry in c)}")
    m = [regression_bonsai["body"][i][0] for i in range(len(regression_bonsai["body"]))
         if regression_bonsai["body"][i] is not None]
    c = [regression_bonsai["body"][i][1] for i in range(len(regression_bonsai["body"]))
         if regression_bonsai["body"][i] is not None]
    if len(m) > 0 or len(c) > 0:
        print(f"  For body videos: bonsai factors: {', '.join(str(entry) for entry in m)}; "
              f"bonsai offsets: {', '.join(str(entry) for entry in c)}")

    return regression_bonsai


def check_videos_swapped(data, n_frames_body, n_frames_eye, nt_body, nt_eye):
    idx_valid = [i for i in range(len(nt_eye)) if nt_eye[i] is not None or nt_body[i] is not None]
    if len(idx_valid) > 0:
        n_frames_body = np.array(n_frames_body)[idx_valid]
        n_frames_eye = np.array(n_frames_eye)[idx_valid]
        n_frames = np.concatenate((n_frames_eye, n_frames_body))
        n_TTL_eye = [len(nt_eye[i]) for i in idx_valid]
        n_TTL_body =  [len(nt_body[i]) for i in idx_valid]
        diff_TTL1 = np.sum(np.abs(n_frames - np.array(n_TTL_eye + n_TTL_body)))
        diff_TTL2 = np.sum(np.abs(np.array(n_frames) - np.array(n_TTL_body + n_TTL_eye)))
        if diff_TTL2 < diff_TTL1 * 0.95:
            print("  Warning: eye and body timestamps seem to be swapped.")
            eye = data["eye"]["eye.timestamps.npy"]
            data["eye"]["eye.timestamps.npy"] = data["body"]["body.timestamps.npy"]
            data["body"]["body.timestamps.npy"] = eye


def collect_stim_data(path, exp_folder, is_atropine, rec_durations):
    stim_data = {}
    for protocol in protocols:
        intv_file = os.path.join(path, exp_folder, f"recording.{protocol}_intervals.npy")
        if not os.path.exists(intv_file):
            continue

        stim_data["protocol"] = protocol
        dur = np.load(intv_file)[0, 1]
        rec_durations.append(dur)

        files = glob.glob(os.path.join(path, exp_folder, f"{protocol}*"))
        if files:
            print(f"    Stimulus: {protocol}")
            for file_path in files:
                file_name = os.path.basename(file_path)
                if file_name.endswith(".csv"):
                    stim_data[file_name] = np.loadtxt(file_path, delimiter=",", dtype=str)
                else:
                    stim_data[file_name] = np.load(file_path)

            if is_atropine:
                key = f"{protocol}Exp.description.npy"
                if key in stim_data.keys():
                    stim_data[key] = stim_data[key] + "Atropine"
                else:
                    stim_data[key] = "Atropine"

        break  # only one stimulus protocol per experiment
    return stim_data


def select_video_times(video_name, data_video, data_time,
                       num_exp, rec_durations, regression_bonsai):
    t_arduino = data_time["time_arduino.npy"]
    nframes_cumsum = np.cumsum(data_video[f"{video_name}.nframes.npy"])
    times = np.ones((nframes_cumsum[-1], 1)) * np.nan
    source = np.zeros((nframes_cumsum[-1], 1))  # 0: invalid, 1: TTL, 2: added to TTL, 3: bonsai log, 4: added to bonsai log
    exp_start = 0
    for i in range(num_exp):
        times_TTL = data_video[f"{video_name}.timestamps.npy"][i]
        n = data_video[f"{video_name}.nframes.npy"][i]
        use_bonsai = False
        idx_start = 0 if i == 0 else nframes_cumsum[i - 1]

        # (0) No TTL pulses
        if times_TTL is None or len(times_TTL) == 0:
            use_bonsai = True
        # (1) If TTL pulses from arduino match number of video frames, take those times
        elif len(times_TTL) == n:
            times[idx_start:nframes_cumsum[i]] = times_TTL + exp_start
            source[idx_start:nframes_cumsum[i]] = np.ones((n, 1))
        # (2) If TTL pulses from arduino closely match number of video frames, and TTL pulses clearly start after 0 or
        # end before recording, take TTL pulses and add missing time points
        elif abs(len(times_TTL) - n) < 20:
            source[idx_start:nframes_cumsum[i]] = np.ones((n, 1))
            dt = np.median(np.diff(times_TTL, axis=0))
            n_missing = n - len(times_TTL)
            if n_missing < 0:
                if n_missing > -5:
                    print(f"    Warning: {video_name} video has {-n_missing} extra TTLs. -> delete at end")
                    times[idx_start:nframes_cumsum[i]] = times_TTL[:n] + exp_start
                else:
                    use_bonsai = True
            elif times_TTL[0] > t_arduino[i][0] + dt:  # video started after recording -> add time points to the end
                extra_times = times_TTL[-1] + (np.arange(1, n_missing + 1) * dt).reshape(-1, 1)
                times[idx_start:nframes_cumsum[i]] = np.vstack((times_TTL, extra_times)) + exp_start
                source[nframes_cumsum[i] - n_missing: nframes_cumsum[i]] = np.ones((n_missing, 1)) * 2
            elif times_TTL[-1] < t_arduino[i][-1] - dt:  # video ended before end of recording -> add time points to the start
                extra_times = times_TTL[0] - (np.arange(-n_missing, 0) * dt).reshape(-1, 1)
                times[idx_start:nframes_cumsum[i]] = np.vstack((extra_times, times_TTL)) + exp_start
                source[:n_missing] = np.ones((n_missing, 1)) * 2
            else:
                use_bonsai = True
        else:
            use_bonsai = True

        # (3) Use timestamps recorded by bonsai log file (if good enough)
        if use_bonsai:
            times_TTL = data_video[f"{video_name}.timestamps_bonsai.npy"][i]
            # conditions:
            # (1) MSE from linear regression of stimulus times is small
            # Remove: (1) number of timestamps from bonsai log file is similar to number of timestamps from nidaq (up to 100 ms difference);
            # (2) number of bonsai video times is similar to number video frames (up to 5 frames difference)
            if regression_bonsai["stimuli"][i] is not None:
                (m, c, mse) = regression_bonsai["stimuli"][i]
            else:
                # find those experiments with low MSE
                idx_valid = np.array([i for i in range(len(regression_bonsai["stimuli"]))
                             if
                             regression_bonsai["stimuli"][i] is not None and regression_bonsai["stimuli"][i][2] < 1e-2])
                m = []
                c = []
                mse = []
                for j in idx_valid:
                    m.append(regression_bonsai["stimuli"][j][0])
                    c.append(regression_bonsai["stimuli"][j][1])
                    mse.append(regression_bonsai["stimuli"][j][2])
                m = np.mean(np.array(m))
                c = np.mean(np.array(c))
                mse = np.mean(np.array(mse))

            if mse < 1e-2 and n - len(times_TTL) <= 5:
                source[idx_start:nframes_cumsum[i]] = np.ones((n, 1)) * 3
                if len(times_TTL) < n:
                    dt = np.median(np.diff(times_TTL, axis=0))
                    extra_times = times_TTL[-1] + (np.arange(1, n - len(times_TTL) + 1) * dt).reshape(-1, 1)
                    t = np.vstack((times_TTL, extra_times)) * m + c
                    times[idx_start:nframes_cumsum[i]] = t + exp_start
                    n_missing = n - len(times_TTL)
                    source[nframes_cumsum[i] - n_missing: nframes_cumsum[i]] = np.ones((n_missing, 1)) * 4
                else:
                    t = times_TTL * m + c
                    times[idx_start:nframes_cumsum[i]] = t + exp_start
            else:
                print(f"  Warning: could not find good timestamps for {video_name} in experiment {i + 1}.")

        exp_start += rec_durations[i] + GAP

    return times, source


def preprocess(config: dict, datasets: pd.DataFrame):
    # Directory for all run logs.
    log_dir = config["directories"]["output"].format(Name="ALL", Date="ALL")

    session_log_path = general.make_incremental_log_path(log_dir, base_name="main_combine", ext=".txt")
    log_file, tee = general.make_logger(session_log_path)
    try:
        with redirect_stdout(tee), redirect_stderr(tee):
            for i in range(len(datasets)):
                # if i < 35:
                #     continue
                if not datasets.loc[i]["Process"]:
                    continue

                print(f"\n\nProcessing {datasets.loc[i]['Name']} {datasets.loc[i]['Date']}...")
                paths = general.make_paths(datasets.loc[i], config["directories"])
                combine_metadata(paths["output"], datasets.loc[i]["Atropine"])
    finally:
        log_file.close()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset_list(path):
    df = pd.read_csv(path, dtype={
        "Name": str,
        "Date": str,
        "Atropine": str,
        "Process": bool
    })
    df["Atropine"] = df["Atropine"].apply(ast.literal_eval)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess TwoP meta-data')
    parser.add_argument('--config', type=str,
                        required=False, default='preprocess.yaml',
                        help='Path to preprocess.yaml')
    return parser.parse_args()


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