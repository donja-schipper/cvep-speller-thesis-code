import os
import mne
from mnelab.io.xdf import read_raw_xdf as read_raw
import numpy as np
import pyxdf
import pyntbci


KEYS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "dot", "comma", "question", "exclam", "space", "backspace",
]

TRIAL_TIME = 4.2  # duration of a trial in seconds
FS = 120  # target sampling frequency of the EEG in Hertz
PR = 60  # stimulus presentation rate in Hertz
L_FREQ = 6.00  # lower cutoff for the high-pass filter in Hertz
H_FREQ = 21.00  # higher cutof for the low-pass filter in Hertz


def main(i_subject, data_dir):
    subject = f"sub-{i_subject:03d}"
    print(subject)

    # Create output folder
    if not os.path.exists(os.path.join(data_dir, "derivatives", subject)):
        os.makedirs(os.path.join(data_dir, "derivatives", subject))

    # Set stimuli
    V = pyntbci.stimulus.shift(pyntbci.stimulus.make_m_sequence(), stride=1)[::2, :]
    V = np.repeat(V, int(FS / PR), axis=1).astype("uint8")

    # XDF file name
    fn = os.path.join(data_dir, "sourcedata", subject, "ses-001", "eeg", f"{subject}_ses-001_task-cvep_run-001_eeg.xdf")

    # Load EEG
    streams = pyxdf.resolve_streams(fn)
    names = [stream["name"] for stream in streams]
    raw = read_raw(fn,
                   stream_ids=[streams[names.index("BioSemi")]["stream_id"]],
                   marker_ids=[streams[names.index("marker-stream")]["stream_id"]])
    raw = raw.drop_channels([f"EX{i}" for i in range(1, 9)] + [f"AIB{i}" for i in range(1, 33)] + ["Trig1"])

    # Read events
    cue_events, cue_event_id = mne.events_from_annotations(raw, regexp="start_cue*", verbose=False)
    assert cue_events.shape[0] == 384, f"\tFound more/less than 384 cue events: {cue_events.shape[0]}"
    trial_events, trial_event_id = mne.events_from_annotations(raw, regexp="start_trial*", verbose=False)
    assert trial_events.shape[0] == 384, f"\tFound more/less than 384 trial events: {trial_events.shape[0]}"

    # Filtering
    raw = raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)

    # Slicing
    # N.B. add 0.5 sec pre- and post-trial to capture filtering artefacts of downsampling (removed later on)
    # N.B. Use the largest trial time (samples are cut away later)
    epo = mne.Epochs(raw, events=trial_events, tmin=-0.5, tmax=TRIAL_TIME + 0.5, baseline=None, picks="eeg",
                     preload=True, proj=False, verbose=False)

    # Resampling
    # N.B. Downsampling is done after slicing to maintain accurate stimulus timing
    epo = epo.resample(sfreq=FS, verbose=False)

    # Extract data
    X = epo.get_data(tmin=0, tmax=TRIAL_TIME).astype("float32")
    print(f"\tFound {X.shape[0]} trials in the EEG")

    # Load labels and conditions
    keys = []
    conditions = []
    contrasts = []
    for i in range(cue_events.shape[0]):
        event = cue_events[i, 2]
        marker = list(cue_event_id.keys())[list(cue_event_id.values()).index(event)]
        cue, key_idx, key, condition, contrast = marker.replace(" ", "").split(";")
        keys.append(key.split("=")[1])
        conditions.append(condition.split("=")[1])
        contrasts.append(contrast.split("=")[1])

    # Keys to labels
    y = np.array([KEYS.index(key) for key in keys]).astype("uint8")
    print(f"\tFound {y.size} labels")

    # Save conditions to separate files
    print("\tFound the following conditions:", set(conditions))
    print("\tFound the following contrasts:", set(contrasts))
    for condition in set(conditions):
        for contrast in set(contrasts):
            # Select condition-contrast
            idx = np.array([(x == condition) and (y == contrast) for x, y in zip(conditions, contrasts)])
            X_ = X[idx, :, :]
            y_ = y[idx]

            # Print summary
            print(f"\tCondition-{condition} contrast-{contrast}")
            print("\t\tX:", X_.shape, X_.dtype)
            print("\t\ty:", y_.shape, y_.dtype)
            print("\t\tV:", V.shape, V.dtype)

            # Save data
            fn = os.path.join(data_dir, "derivatives", subject, f"{subject}_cvep_{condition}_{contrast}.npz")
            np.savez(fn, X=X_, y=y_, V=V, fs=FS)


if __name__ == "__main__":
    data_dir = r"C:\Users\donja\Desktop\Thesis\Data"
    for i_subject in range(2, 6):
        main(i_subject, data_dir)
