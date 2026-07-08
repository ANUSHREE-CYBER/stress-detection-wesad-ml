import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR
from src.constants import LABEL_NAMES, META_COLS

# ── CONFIG ──────────────────────────────────────────────────────────────
PROCESSED_PATH = DATA_DIR
FS = 700  # Hz

# ── LOAD WINDOWS ─────────────────────────────────────────────────────────
print("Loading windows...")
all_windows = np.load(
    os.path.join(PROCESSED_PATH, "all_windows.npy"),
    allow_pickle=True
)
print(f"Loaded {len(all_windows)} windows")

# ── ECG / HRV FEATURES ───────────────────────────────────────────────────
def extract_hrv_features(ecg, fs=700):
    # Detect R-peaks
    peaks, _ = find_peaks(ecg, distance=fs*0.4, height=np.mean(ecg))
    
    if len(peaks) < 3:
        return None
    
    # RR intervals in milliseconds
    rr = np.diff(peaks) / fs * 1000
    
    features = {
        # Time domain HRV
        'hrv_mean_rr'   : np.mean(rr),
        'hrv_std_rr'    : np.std(rr),
        'hrv_rmssd'     : np.sqrt(np.mean(np.diff(rr)**2)),
        'hrv_pnn50'     : np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100,
        'hrv_min_rr'    : np.min(rr),
        'hrv_max_rr'    : np.max(rr),
        'hrv_range_rr'  : np.max(rr) - np.min(rr),
        'hrv_mean_hr'   : 60000 / np.mean(rr),  # BPM
        
        # Frequency domain (simple LF/HF ratio via RR stats)
        'hrv_cv'        : np.std(rr) / np.mean(rr),  # Coefficient of variation
        'hrv_skew_rr'   : skew(rr),
        'hrv_kurt_rr'   : kurtosis(rr),
        'hrv_num_peaks' : len(peaks),
    }
    return features

# ── EDA FEATURES ─────────────────────────────────────────────────────────
def extract_eda_features(eda):
    features = {
        'eda_mean'    : np.mean(eda),
        'eda_std'     : np.std(eda),
        'eda_min'     : np.min(eda),
        'eda_max'     : np.max(eda),
        'eda_range'   : np.max(eda) - np.min(eda),
        'eda_skew'    : skew(eda),
        'eda_kurt'    : kurtosis(eda),
        # Slope (trend)
        'eda_slope'   : np.polyfit(np.arange(len(eda)), eda, 1)[0],
    }
    return features

# ── RESPIRATION FEATURES ─────────────────────────────────────────────────
def extract_resp_features(resp):
    features = {
        'resp_mean'   : np.mean(resp),
        'resp_std'    : np.std(resp),
        'resp_min'    : np.min(resp),
        'resp_max'    : np.max(resp),
        'resp_range'  : np.max(resp) - np.min(resp),
        'resp_skew'   : skew(resp),
        'resp_kurt'   : kurtosis(resp),
        'resp_slope'  : np.polyfit(np.arange(len(resp)), resp, 1)[0],
    }
    return features

# ── EMG FEATURES ─────────────────────────────────────────────────────────
def extract_emg_features(emg):
    features = {
        'emg_mean'    : np.mean(np.abs(emg)),
        'emg_std'     : np.std(emg),
        'emg_max'     : np.max(np.abs(emg)),
        'emg_rms'     : np.sqrt(np.mean(emg**2)),
        'emg_skew'    : skew(emg),
        'emg_kurt'    : kurtosis(emg),
    }
    return features

# ── EXTRACT ALL FEATURES ─────────────────────────────────────────────────
print("Extracting features from each window...")
feature_rows = []
skipped = 0

# NaN/inf tracking — constant/zero-variance windows can produce NaN (skew,
# kurtosis, std/mean) or inf (division by zero) in the stats above.
windows_affected = 0
total_nan = 0
total_inf = 0

for i, window in enumerate(all_windows):
    if i % 100 == 0:
        print(f"  Processing window {i}/{len(all_windows)}...")

    hrv  = extract_hrv_features(window['ecg'], fs=FS)
    if hrv is None:
        skipped += 1
        continue

    eda  = extract_eda_features(window['eda'])
    resp = extract_resp_features(window['resp'])
    emg  = extract_emg_features(window['emg'])

    row = {}
    row.update(hrv)
    row.update(eda)
    row.update(resp)
    row.update(emg)

    # ── NaN/inf handling (numeric feature values only) ────────────────────
    feat_keys   = list(row.keys())
    feature_row = np.array([row[k] for k in feat_keys], dtype=float)
    nan_count   = int(np.isnan(feature_row).sum())
    inf_count   = int(np.isinf(feature_row).sum())
    if nan_count > 0 or inf_count > 0:
        print(f"Window {i} (subject {window['subject_id']}): "
              f"{nan_count} NaN, {inf_count} inf values replaced with 0.0")
        windows_affected += 1
        total_nan += nan_count
        total_inf += inf_count
    feature_row = np.nan_to_num(feature_row, nan=0.0, posinf=0.0, neginf=0.0)
    for k, v in zip(feat_keys, feature_row):
        row[k] = v

    # Metadata columns — carried through, NOT model features
    row['label']      = window['label']
    row['subject_id'] = window['subject_id']
    feature_rows.append(row)

print(f"\nSkipped {skipped} windows (too few R-peaks)")
print(f"Final feature rows: {len(feature_rows)}")

# ── NaN/inf SUMMARY ──────────────────────────────────────────────────────
if feature_rows:
    pct = windows_affected / len(feature_rows) * 100
    print(f"\nNaN/inf cleanup: {windows_affected} of {len(feature_rows)} windows "
          f"({pct:.1f}%) had values replaced with 0.0 "
          f"— {total_nan} NaN and {total_inf} inf total")
else:
    print("\nNaN/inf cleanup: no feature rows produced")

# ── SAVE FEATURES ────────────────────────────────────────────────────────
df = pd.DataFrame(feature_rows)
print(f"\nFeature matrix shape: {df.shape}")
feature_cols = [c for c in df.columns if c not in META_COLS]
print(f"\nFeatures extracted ({len(feature_cols)} total):")
print(feature_cols)

print(f"\nLabel distribution:")
print(df['label'].value_counts().rename(LABEL_NAMES))

output_csv = os.path.join(PROCESSED_PATH, "features.csv")
df.to_csv(output_csv, index=False)
print(f"\nSaved to {output_csv}")
print("\n✅ Feature extraction complete!")