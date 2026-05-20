import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ── CONFIG ──────────────────────────────────────────────────────────────
DATASET_PATH  = r"D:\WESAD Dataset\WESAD"
OUTPUT_PATH   = r"D:\STRESS DETECTION\data\processed"
SUBJECTS      = ['S2','S3','S4','S5','S6','S7','S8','S9',
                 'S10','S11','S13','S14','S15','S16','S17']
KEEP_LABELS   = {1: 0, 2: 1, 3: 2}   # baseline=0, stress=1, amusement=2
FS_CHEST      = 700                   # Hz
WINDOW_SEC    = 60                    # 60-second windows
STEP_SEC      = 30                    # 30-second overlap (50% overlap)
WINDOW_SAMP   = WINDOW_SEC * FS_CHEST # samples per window = 42000
STEP_SAMP     = STEP_SEC  * FS_CHEST  # step size = 21000

# ── BANDPASS FILTER ──────────────────────────────────────────────────────
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=700, order=4):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal.flatten())

# ── LOAD ONE SUBJECT ─────────────────────────────────────────────────────
def load_subject(subject_id):
    pkl_path = os.path.join(DATASET_PATH, subject_id, f"{subject_id}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ecg  = data['signal']['chest']['ECG'].flatten()
    eda  = data['signal']['chest']['EDA'].flatten()
    resp = data['signal']['chest']['Resp'].flatten()
    emg  = data['signal']['chest']['EMG'].flatten()
    labels = data['label'].flatten()

    # Filter ECG
    ecg_filtered = bandpass_filter(ecg, lowcut=0.5, highcut=40.0, fs=FS_CHEST)

    return ecg_filtered, eda, resp, emg, labels

# ── SLIDING WINDOW ───────────────────────────────────────────────────────
def sliding_windows(ecg, eda, resp, emg, labels):
    windows = []
    n = len(ecg)
    for start in range(0, n - WINDOW_SAMP, STEP_SAMP):
        end = start + WINDOW_SAMP
        window_labels = labels[start:end]

        # Find majority label in window
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        # Only keep windows with clean labels (1, 2, or 3)
        if majority_label not in KEEP_LABELS:
            continue

        # Check window is mostly one label (>80% purity)
        purity = np.max(counts) / len(window_labels)
        if purity < 0.8:
            continue

        windows.append({
            'ecg' : ecg [start:end],
            'eda' : eda [start:end],
            'resp': resp[start:end],
            'emg' : emg [start:end],
            'label': KEEP_LABELS[majority_label]
        })
    return windows

# ── MAIN LOOP ────────────────────────────────────────────────────────────
print("Processing subjects...")
all_windows = []
subject_counts = {}

for subj in SUBJECTS:
    print(f"  Loading {subj}...", end=" ")
    try:
        ecg, eda, resp, emg, labels = load_subject(subj)
        windows = sliding_windows(ecg, eda, resp, emg, labels)
        all_windows.extend(windows)
        subject_counts[subj] = len(windows)
        print(f"{len(windows)} windows")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nTotal windows across all subjects: {len(all_windows)}")

# ── LABEL DISTRIBUTION ───────────────────────────────────────────────────
label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}
all_labels  = [w['label'] for w in all_windows]
print("\nLabel distribution:")
for lbl, name in label_names.items():
    count = all_labels.count(lbl)
    print(f"  {name}: {count} windows ({count/len(all_labels)*100:.1f}%)")

# ── SAVE WINDOWS ─────────────────────────────────────────────────────────
print("\nSaving processed windows...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
np.save(os.path.join(OUTPUT_PATH, "all_windows.npy"),
        np.array(all_windows, dtype=object))
print(f"Saved to {OUTPUT_PATH}\\all_windows.npy")

print("\n✅ Data loading complete!")