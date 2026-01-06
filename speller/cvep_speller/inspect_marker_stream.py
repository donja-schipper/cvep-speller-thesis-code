"""
Adapted from https://github.com/thijor/speller.
"""
import os
import pyxdf
from collections import Counter
import re

DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
SUBJECT = "P003"
SESSION = "S001"
RUN = "001"
TASK = "Default"

reset_to_zero = True

# Load streams
fn = os.path.join(DATA_DIR, f"sub-{SUBJECT}", f"ses-{SESSION}", "eeg",
                  f"sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_run-{RUN}_eeg.xdf")
streams = pyxdf.load_xdf(fn)[0]

# Extract stream names
names = [stream["info"]["name"][0] for stream in streams]
print("Available streams:", ", ".join(names))

# Marker stream
marker_stream = streams[names.index("marker-stream")]
if reset_to_zero:
    marker_stream["time_stamps"] -= marker_stream["time_stamps"][0]

# Counters
cue_events = 0
trial_count = 0
rest_count = 0
condition_counts = Counter()
contrast_counts = Counter()
condition_contrast = Counter()
key_counts = Counter()
key_cond_cont = Counter()

# Regex
re_key  = re.compile(r"key=([^;]+)")
re_cond = re.compile(r"condition=([^;]+)")
re_cont = re.compile(r"contrast=(\d+)")

for t, m in zip(marker_stream["time_stamps"], marker_stream["time_series"]):
    marker = m[0]

    if marker.startswith("start_cue"):
        cue_events += 1

        key_m  = re_key.search(marker)
        cond_m = re_cond.search(marker)
        cont_m = re_cont.search(marker)

        key  = key_m.group(1)       if key_m  else None
        cond = cond_m.group(1)      if cond_m else None
        cont = int(cont_m.group(1)) if cont_m else None

        if key is not None:
            key_counts[key] += 1
        if cond is not None:
            condition_counts[cond] += 1
        if cont is not None:
            contrast_counts[cont] += 1
        if cond is not None and cont is not None:
            condition_contrast[(cond, cont)] += 1
        if key is not None and cond is not None and cont is not None:
            key_cond_cont[(key, cond, cont)] += 1

    elif marker.startswith("start_trial"):
        trial_count += 1

    elif marker.startswith("start_rest"):
        rest_count += 1

    print(f"{t}\t{marker}")

sum_keys = sum(key_counts.values())
sum_cc   = sum(condition_contrast.values())
if not (cue_events == sum_keys == sum_cc):
    print("\n[WARN] Mismatch between cue-derived totals:")
    print(f"  cue_events={cue_events}, sum_keys={sum_keys}, sum_condition×contrast={sum_cc}")

print("\n===== SUMMARY =====")
print(f"Total trials: {trial_count}")
print(f"Total rest periods: {rest_count}\n")
print("\nContrast totals (from start_cue):")
for cont, n in sorted(contrast_counts.items()):
    print(f"  {cont:3d}%: {n}")

print("Trials per condition and contrast (from start_cue):")
for (cond, cont), n in sorted(condition_contrast.items()):
    print(f"  {cond:8s} contrast {cont:3d}% → {n} trials")

print("\nKey cue counts (from start_cue):")
for key, n in sorted(key_counts.items()):
    print(f"  {key:10s}: {n}")

print("\nCondition totals (from start_cue):")
for cond, n in sorted(condition_counts.items()):
    print(f"  {cond:8s}: {n}")

print("\nContrast totals (from start_cue):")
for cont, n in sorted(contrast_counts.items()):
    print(f"  {cont:3d}%: {n}")

# Check per key whether they had all condition×contrast combos once
print("\nPer-key condition×contrast coverage:")
cells = sorted(condition_contrast.keys())  # all (condition, contrast) pairs seen overall

for key in sorted(key_counts.keys()):
    missing = []
    extra = []

    for cond, cont in cells:
        n = key_cond_cont.get((key, cond, cont), 0)
        if n == 0:
            missing.append(f"{cond}-{cont}%")
        elif n > 1:
            extra.append(f"{cond}-{cont}% (n={n})")

    if not missing and not extra:
        print(f"{key:10s}: OK (exactly one trial in every condition×contrast)")
    else:
        print(f"{key:10s}:")
        if missing:
            print(f"  missing: {', '.join(missing)}")
        if extra:
            print(f"  extra:   {', '.join(extra)}")