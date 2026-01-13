# Decoding uses rCCA implementation from pyntbci (https://github.com/thijor/pyntbci; Thielen et al., 2015).

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pyntbci.classifiers import rCCA
from sklearn.model_selection import ShuffleSplit

BAD_CHANNELS = {"sub-005": [24]}

def apply_channel_selection(subject, X):
    if subject in BAD_CHANNELS:
        for ch in sorted(BAD_CHANNELS[subject], reverse=True):
            X = np.delete(X, ch, axis=1)

    return X

def compute_cv_accuracy(clf, X, y, V, fs, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for train_idx, test_idx in cv.split(X):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        accs.append((y_pred == y[test_idx]).mean())
    return float(np.mean(accs))

def compute_detection_time(X, fs):
    n_samples = X.shape[-1]
    T = n_samples / float(fs)
    return float(T)

def compute_itr(P, N, T):
    if P <= 0:
        return 0.0
    if P >= 1.0:
        B = np.log2(N)
    else:
        B = (
            np.log2(N)
            + P * np.log2(P)
            + (1 - P) * np.log2((1 - P) / (N - 1))
        )
    ITR = B * (60.0 / T)
    return float(ITR)

def parse_condition(fn):
    base = os.path.basename(fn).replace(".npz", "")
    parts = base.split("_")  # ['sub-001', 'cvep', 'classic', '100']
    subject = parts[0]
    stim_type = parts[2]
    contrast = int(parts[3])
    return subject, stim_type, contrast


# In[5]:

def compute_metrics_for_file(fn):
    data = np.load(fn)
    X = data["X"]
    y = data["y"]
    V = data["V"]
    fs = int(data["fs"])

    filename = os.path.basename(fn)
    subject, _, _ = parse_condition(filename)

    X = apply_channel_selection(subject, X)

    # # detect sub-005 
    # if filename.startswith("sub-005"):
    #     # B25 is channel index 24 - broken electrode
    #     bad_ch = 24
    #     X = np.delete(X, bad_ch, axis=1) 

    clf = rCCA(stimulus=V, fs=fs, event='refe', onset_event=True, encoding_length=0.3)

    acc = compute_cv_accuracy(clf, X, y, V, fs)
    T = compute_detection_time(X, fs)
    N = len(np.unique(y))
    itr = compute_itr(acc, N, T)
    return acc, T, itr

def build_metrics_dataframe(base_dir):
    pattern= os.path.join(base_dir, "sub-*", "sub-*_cvep_*.npz")
    all_files = sorted(glob.glob(pattern))
    results = []
    for fn in all_files:
        subject, stim_type, contrast = parse_condition(fn)
        acc, T, itr = compute_metrics_for_file(fn)
        results.append({
            "subject": subject,
            "file": os.path.basename(fn),
            "stim_type": stim_type,
            "contrast": contrast,
            "accuracy": acc,
            "decision_time": T,
            "itr": itr,
        })
    df = pd.DataFrame(results)
    return df


# In[13]:


def permutation_pvalue_above_chance(fn, true_acc, n_perm=100):
    """Permutation test for a single condition."""
    data = np.load(fn)
    X = data["X"]
    y = data["y"]
    V = data["V"]
    fs = int(data["fs"])
    
    filename = os.path.basename(fn)
    subject, _, _ = parse_condition(filename)

    X = apply_channel_selection(subject, X)

    perm_accs = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = np.random.permutation(y)
        clf = rCCA(stimulus=V, fs=fs, event='refe', onset_event=True, encoding_length=0.3,)
        perm_accs[i] = compute_cv_accuracy(clf, X, y_perm, V, fs)

    p_value = np.mean(perm_accs >= true_acc)
    return p_value, perm_accs

def run_permutation_tests(df, base_dir, n_perm=100):
    perm_results = []

    for _, row in df.iterrows():
        filename = row["file"]
        subject = row["subject"]
        fn = os.path.join(base_dir, subject, filename) 

        true_acc = float(row["accuracy"])

        p_value, perm_accs = permutation_pvalue_above_chance(
            fn, true_acc, n_perm=n_perm
        )

        perm_results.append({
            "subject": subject,
            "file": filename,
            "stim_type": row["stim_type"],
            "contrast": row["contrast"],
            "true_accuracy": true_acc,
            "p_above_chance": p_value,
            "perm_accs": perm_accs,
        })

    df_perm = pd.DataFrame(perm_results)

    # Merge p-values back into the original metrics df
    df_with_p = df.merge(
        df_perm[["subject", "file", "p_above_chance"]],
        on=["subject", "file"],
        how="left",
    )

    return df_with_p, df_perm



def compute_accuracy_for_condition(fn):
    data = np.load(fn)
    X = data["X"]
    y = data["y"]
    V = data["V"]
    fs = int(data["fs"])

    filename = os.path.basename(fn)
    subject, _, _ = parse_condition(filename)

    X = apply_channel_selection(subject, X)
    
    clf = rCCA(stimulus=V, fs=fs, event='refe', onset_event=True, encoding_length=0.3)
    acc = compute_cv_accuracy(clf, X, y, V, fs)
    return acc, X, y, V, fs


def permutation_test_classic_vs_grating(fn_classic, fn_grating, n_perm=100):
    # Load both datasets
    acc_classic, X_c, y_c, V_c, fs = compute_accuracy_for_condition(fn_classic)
    acc_grating, X_g, y_g, V_g, fs = compute_accuracy_for_condition(fn_grating)

    # Real difference
    d_real = acc_grating - acc_classic

    # Concatenate trials from both conditions - 
    X = np.concatenate([X_c, X_g], axis=0)
    y = np.concatenate([y_c, y_g], axis=0)

    # Labels indicating condition (0 = classic, 1 = grating)
    cond_labels = np.array([0]*len(y_c) + [1]*len(y_g))

    # Build the null distribution
    null_diffs = np.zeros(n_perm)

    for i in range(n_perm):
        # Shuffle condition labels
        shuffled = np.random.permutation(cond_labels)

        # Split trials according to shuffled labels
        X_c_shuf = X[shuffled == 0]
        y_c_shuf = y[shuffled == 0]

        X_g_shuf = X[shuffled == 1]
        y_g_shuf = y[shuffled == 1]

        # Recompute accuracy for shuffled classic
        clf_c = rCCA(stimulus=V_c, fs=fs, event='refe', onset_event=True, encoding_length=0.3)
        acc_c_shuf = compute_cv_accuracy(clf_c, X_c_shuf, y_c_shuf, V_c, fs)

        # Recompute accuracy for shuffled grating
        clf_g = rCCA(stimulus=V_g, fs=fs, event='refe', onset_event=True, encoding_length=0.3)
        acc_g_shuf = compute_cv_accuracy(clf_g, X_g_shuf, y_g_shuf, V_g, fs)

        null_diffs[i] = acc_g_shuf - acc_c_shuf

    # p-value = proportion of shuffled differences >= real difference
    p_value = np.mean(null_diffs >= d_real)

    return d_real, p_value, null_diffs

def learning_curve_for_file(fn, train_fracs=None, n_repeats=10):
    if train_fracs==None:
        train_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]

    data = np.load(fn)
    X = data["X"]
    y = data["y"]
    V = data["V"]
    fs = int(data["fs"])
    
    subject, _, _ = parse_condition(fn)

    X = apply_channel_selection(subject, X)

    n_trials = X.shape[0]
    train_fracs = np.array(train_fracs)

    acc_means = []

    for f in train_fracs:
        # random splits with given training fraction
        splitter = ShuffleSplit(
            n_splits=n_repeats,
            train_size=f,
            test_size=None,
            random_state=0
        )

        accs = []

        for train_idx, test_idx in splitter.split(X):
            clf = rCCA(stimulus=V, fs=fs, event='refe', onset_event=True, encoding_length=0.3)

            # fit on training subset
            clf.fit(X[train_idx], y[train_idx])

            # test on remaining
            y_pred = clf.predict(X[test_idx])
            accs.append((y_pred == y[test_idx]).mean())

        acc_means.append(np.mean(accs))

    return train_fracs, np.array(acc_means), n_trials

def compute_learning_curves(base_dir, train_fracs=None, n_repeats=10):
    pattern= os.path.join(base_dir, "sub-*", "sub-*_cvep_*.npz")
    all_files = sorted(glob.glob(pattern))
    if train_fracs is None:
        train_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    curves = []

    for fn in all_files:
        subject, stim_type, contrast = parse_condition(fn)

        fracs, accs, n_trials = learning_curve_for_file(fn, train_fracs=train_fracs, n_repeats=n_repeats,)

        curves.append({
            "subject": subject,
            "file": os.path.basename(fn),
            "stim_type": stim_type,
            "contrast": contrast,
            "train_fracs": fracs,
            "accs": accs,
            "n_trials": n_trials,
        })

    return curves

def rcca_snr_for_file(fn, n_splits=5):
    data = np.load(fn)
    X = data["X"]      # (trials, channels, samples)
    y = data["y"]      # (trials,)
    V = data["V"]      # stimulus codes
    fs = int(data["fs"])

    subject, _, _ = parse_condition(fn)

    X = apply_channel_selection(subject, X)

    cv = KFold(n_splits=n_splits, shuffle=False)

    margins = []

    for train_idx, test_idx in cv.split(X):
        clf = rCCA(stimulus=V, fs=fs, event="refe", onset_event=True, encoding_length=0.3)

        # fit on training data
        clf.fit(X[train_idx], y[train_idx])

        scores = clf.decision_function(X[test_idx])

        # compute margin for each test trial
        for i, idx in enumerate(test_idx):
            true_label = y[idx]
            s_true = scores[i, true_label]
            s_others = np.delete(scores[i, :], true_label)
            margin = s_true - np.max(s_others)
            margins.append(margin)

    margins = np.array(margins)
    return float(np.mean(margins)), float(np.std(margins))


def compute_evoked_snr_for_file(fn, ch_idx=None):
    data = np.load(fn)
    X = data["X"]
    y = data["y"] 
    fs = int(data["fs"])

    filename = os.path.basename(fn)

    subject, _, _ = parse_condition(filename)

    X = apply_channel_selection(subject, X)

    # Select channels
    if ch_idx is not None:
        X = X[:, ch_idx, :]

    # mean over trials
    mean_over_trials = X.mean(axis=0)

    # variance over trials
    var_over_trials = X.var(axis=0, ddof=1)

    # Signal power
    signal_power = np.mean(mean_over_trials**2, axis=-1)

    # Noise power
    noise_power = np.mean(var_over_trials, axis=-1)

    eps = 1e-12
    #snr_per_channel = 10.0 * np.log10((signal_power + eps) / (noise_power + eps))
    snr_per_channel = (signal_power + eps) / (noise_power + eps)

    # Average across channels
    snr_linear = float(np.mean(snr_per_channel))
    return snr_linear


def build_evoked_snr_dataframe(base_dir, ch_idx=None):
    pattern = os.path.join(base_dir, "sub-*", "sub-*_cvep_*.npz")
    all_files = sorted(glob.glob(pattern))
    results = []

    for fn in all_files:
        subject, stim_type, contrast = parse_condition(fn)
        snr_linear = compute_evoked_snr_for_file(fn, ch_idx=ch_idx)

        results.append({
            "subject": subject,
            "file": os.path.basename(fn),
            "stim_type": stim_type,
            "contrast": contrast,
            "snr_linear": snr_linear,
        })

    df_snr = pd.DataFrame(results)
    return df_snr