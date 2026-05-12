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

file_names = {
    "time": ["time_nidaq.npy", "time_arduino.npy", "bonsai.ntimestamps.npy"],
    "traces": ["2pCalcium.timestamps.npy", "2pPlanes.delay.npy"],
    "eye": ["eye.timestamps.npy", "eye.timestamps_bonsai.npy", "eye.nframes.npy"],
    "body": ["body.timestamps.npy", "body.timestamps_bonsai.npy", "body.nframes.npy"],
    "wheel": ["wheel.velocity.npy"],
}
protocols = ["gratings", "circles", "fullField", "darkScreen"]


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
                    print(f"    File {file_name} not found in {exp_folder}.")

        exp_num = int(''.join(filter(str.isdigit, exp_folder)))
        is_atropine = exp_num in exp_atropine

        stim_data = {}
        for protocol in protocols:
            intv_file = os.path.join(path, exp_folder, f"recording.{protocol}_intervals.npy")
            if not os.path.exists(intv_file):
                continue

            stim_data["protocol"] = protocol
            dur = np.load(intv_file)[-1,0]
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

        stimuli.append(stim_data)

    # For each experiment: check video time data (eye + body) and decide which times to use if any
    times_eye, source_eye = select_video_times("eye", data["eye"], data["time"], len(exp_folders), rec_durations)
    if not all(np.isnan(times_eye)):
        np.save(os.path.join(path, "eye.timestamps.npy"), times_eye)
        np.save(os.path.join(path, "eye.time_source.npy"), source_eye)

    times_body, source_body = select_video_times("body", data["body"], data["time"], len(exp_folders), rec_durations)
    if not all(np.isnan(times_body)):
        np.save(os.path.join(path, "body.timestamps.npy"), times_body)
        np.save(os.path.join(path, "body.time_source.npy"), source_body)

    # Combine trace times, arduino time, and wheel data
    times_calcium = np.zeros((0,1))
    time_arduino = np.zeros((0,1))
    wheel_velocity = np.zeros((0,1))
    for i in range(len(exp_folders)):
        exp_start = 0 if i == 0 else sum(rec_durations[:i]) + i * GAP
        times_calcium = np.vstack((times_calcium, data["traces"]["2pCalcium.timestamps.npy"][i] + exp_start))
        time_arduino = np.vstack((time_arduino, data["time"]["time_arduino.npy"][i] + exp_start))
        wheel_velocity = np.vstack((wheel_velocity, data["wheel"]["wheel.velocity.npy"][i]))

    np.save(os.path.join(path, "2pCalcium.timestamps.npy"), times_calcium)
    np.save(os.path.join(path, "wheel.timestamps.npy"), time_arduino)
    np.save(os.path.join(path, "wheel.velocity.npy"), wheel_velocity)
    np.save(os.path.join(path, "2pPlanes.delay.npy"), data["traces"]["2pPlanes.delay.npy"][0])  # same for all experiments

    # Combine stim info of several experiments if from same protocol
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

        all_keys.discard("timestamps_bonsai")
        all_keys.discard("protocol")
        for key in all_keys:
            if "intervals.npy" in key:
                intv = np.zeros((0,2))
                for i in idx_exp:
                    exp_start = 0 if i == 0 else sum(rec_durations[:i]) + i * GAP
                    intv = np.vstack((intv, stimuli[i][key] + exp_start))
                np.save(os.path.join(path, key), intv)
            elif "times.npy" in key:
                t = np.zeros((0,1))
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

        rec_intervals = np.zeros((0,2))
        for i in idx_exp:
            exp_start = 0 if i == 0 else sum(rec_durations[:i]) + i * GAP
            intv = np.array([0, float(rec_durations[i])]) + exp_start
            rec_intervals = np.vstack((rec_intervals, intv))
        np.save(os.path.join(path, f"recording.{protocol}_intervals.npy"), rec_intervals)


def select_video_times(video_name, data_video, data_time, num_exp, rec_durations):
    t_arduino = data_time["time_arduino.npy"]
    nframes_cumsum = np.cumsum(data_video[f"{video_name}.nframes.npy"])
    times = np.ones((nframes_cumsum[-1], 1)) * np.nan
    source = np.zeros((nframes_cumsum[-1], 1))  # 0: invalid, 1: TTL, 2: added to TTL, 3: bonsai log, 4: added to bonsai log
    exp_start = 0
    for i in range(num_exp):
        times_exp = data_video[f"{video_name}.timestamps.npy"][i]
        n = data_video[f"{video_name}.nframes.npy"][i]
        use_bonsai = False
        idx_start = 0 if i == 0 else nframes_cumsum[i - 1]

        # (1) If TTL pulses from arduino match number of video frames, take those times
        if len(times_exp) == n:
            times[idx_start:nframes_cumsum[i]] = times_exp + exp_start
            source[idx_start:nframes_cumsum[i]] = np.ones((n, 1))
        # (2) If TTL pulses from arduino closely match number of video frames, and TTL pulses clearly start after 0 or
        # end before recording, take TTL pulses and add missing time points
        elif abs(len(times_exp) - n) < 20:
            source[idx_start:nframes_cumsum[i]] = np.ones((n, 1))
            dt = np.median(np.diff(times_exp, axis=0))
            n_missing = n - len(times_exp)
            if n_missing < 0:
                if n_missing > -5:
                    print(f"    Warning: {video_name} video has {-n_missing} extra TTLs. -> delete at end")
                    times[idx_start:nframes_cumsum[i]] = times_exp[:n] + exp_start
                else:
                    use_bonsai = True
            elif times_exp[0] > t_arduino[i][0] + dt:  # video started after recording -> add time points to the end
                extra_times = times_exp[-1] + (np.arange(1, n_missing + 1) * dt).reshape(-1, 1)
                times[idx_start:nframes_cumsum[i]] = np.vstack((times_exp, extra_times)) + exp_start
                source[nframes_cumsum[i] - n_missing: nframes_cumsum[i]] = np.ones((n_missing, 1)) * 2
            elif times_exp[-1] < t_arduino[i][-1] - dt:  # video ended before end of recording -> add time points to the start
                extra_times = times_exp[0] - (np.arange(-n_missing, 0) * dt).reshape(-1, 1)
                times[idx_start:nframes_cumsum[i]] = np.vstack((extra_times, times_exp)) + exp_start
                source[:n_missing] = np.ones((n_missing, 1)) * 2
            else:
                use_bonsai = True
        else:
            use_bonsai = True

        # (3) Use timestamps recorded by bonsai log file (if good enough)
        if use_bonsai:
            times_exp = data_video[f"{video_name}.timestamps_bonsai.npy"][i]
            bonsai_ntimestamps = data_time["bonsai.ntimestamps.npy"][i]
            nidaq_ntimestamps = len(data_time["time_nidaq.npy"][i])
            # conditions:
            # (1) number of timestamps from bonsai log file is similar to number of timestamps from nidaq (up to 100 ms difference);
            # (2) number of bonsai video times is similar to number video frames (up to 5 frames difference)
            if abs(bonsai_ntimestamps - nidaq_ntimestamps) <= 100 and n - len(times_exp) <= 5:
                source[idx_start:nframes_cumsum[i]] = np.ones((n, 1)) * 3
                if len(times_exp) < n:
                    dt = np.median(np.diff(times_exp, axis=0))
                    extra_times = times_exp[-1] + (np.arange(1, n - len(times_exp) + 1) * dt).reshape(-1, 1)
                    times[idx_start:nframes_cumsum[i]] = np.vstack((times_exp, extra_times)) + exp_start
                    n_missing = n - len(times_exp)
                    source[nframes_cumsum[i] - n_missing: nframes_cumsum[i]] = np.ones((n_missing, 1)) * 4
                else:
                    times[idx_start:nframes_cumsum[i]] = times_exp + exp_start
            else:
                print(f"    Warning: could not find good timestamps for {video_name} in experiment {i + 1}.")

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
                if not datasets.loc[i]["Process"]:
                    continue

                print(f"Processing {datasets.loc[i]['Name']} {datasets.loc[i]['Date']}...")
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