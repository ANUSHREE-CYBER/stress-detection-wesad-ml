---
title: Stress Detection WESAD ML
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# Multimodal Stress Detection System
### Combining Physiological Biosignals + Facial Expressions using ML & Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.8.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

This project is an end-to-end **multimodal stress detection pipeline** that classifies short windows of physiological and/or facial data into three states — **Baseline**, **Stress**, and **Amusement**. It combines two independently-trained models:

- **Biosignal model** — gradient-boosted trees (XGBoost) on 34 hand-crafted features from ECG, EDA, EMG and Respiration signals (WESAD dataset).
- **Vision model** — a MobileNetV2 CNN, transfer-learned from ImageNet, on facial-expression images (FER2013 dataset).

The two models are exposed through an interactive **Streamlit web application** (6 pages), where their probability outputs can also be combined live via a fixed-weight fusion (see [Fusion](#fusion-a-live-proof-of-concept)).

The headline goal of this project is **honest, subject-independent evaluation** — the accuracy numbers below reflect how well the models generalise to *people they have never seen*, not how well they memorise known subjects.

---

## Key Results

All biosignal models are evaluated with **Leave-One-Subject-Out (LOSO) cross-validation**: for each of the 15 subjects, the model is trained on the other 14 and tested on the held-out subject. This guarantees that no window from a test subject ever appears in training, which is the realistic setting for a wearable used on a new person. Accuracy is reported as the **mean ± standard deviation across the 15 held-out subjects** — the std is large on purpose, because per-subject stress signatures genuinely differ.

| Model | Modality | Accuracy (LOSO, mean ± std) | Weighted F1 |
|---|---|---|---|
| SVM | Biosignal | **76.15% ± 13.86%** | 76.84% |
| **XGBoost** (deployed) | Biosignal | **74.57% ± 18.22%** | 74.70% |
| MLP | Biosignal | **72.91% ± 17.69%** | 74.29% |
| MobileNetV2 | Face (held-out test set) | **63.30%** | — |

SVM scores marginally highest, but XGBoost is the **deployed** biosignal model (faster inference, accuracy well within one standard deviation of SVM). All three are trained inside an imbalanced-learn pipeline with **SMOTE applied inside each training fold only** (never on the held-out subject).

### A note on the accuracy correction (a feature, not a bug)

An earlier version of this project reported **~97% accuracy** using random k-fold cross-validation. That split allowed different windows *from the same subject* to land in both the training and test sets, so the models were partly evaluated on people they had already seen — inflating the numbers. This has been corrected to **Leave-One-Subject-Out** cross-validation, which measures generalisation to entirely new people. The honest accuracy lands in the **mid-70s**, which is consistent with published subject-independent results on WESAD. The drop is expected and the corrected methodology is the point.

---

## Fusion: a live proof-of-concept

When a user supplies **both** a biosignal reading and a face image, the app runs each model independently and combines their class probabilities with **fixed heuristic weights — 0.6 biosignal / 0.4 facial**:

```
fused = 0.6 × P(biosignal) + 0.4 × P(face)
```

The biosignal modality is weighted higher because it is the stronger model and a more direct correlate of the autonomic stress response.

**No fusion accuracy is claimed anywhere**, and this is deliberate: WESAD (biosignals) and FER2013 (faces) share **zero subjects**, so there exists no dataset with a genuine biosignal reading *and* a genuine facial expression from the same person at the same moment. Without paired ground truth, any "fusion accuracy" would have to be fabricated. Fusion is therefore presented honestly as a **live multimodal demo** (imagine a wearable + webcam), not a benchmarked model. The weights live in `notebooks/08_fusion.py` and are saved to `models/fusion_config.pkl`.

---

## Architecture & Pipeline

```
WESAD signals ──► 01 load + window ──► 02 features ──► 04 train (LOSO) ──► models/*.pkl ─┐
                                          │                                              ├─► app/app.py
                                          └──► 03 visualize, 05 cluster                  │   (Streamlit)
FER2013 images ─► 06 preprocess ──► 07 train CNN ──► 07b fine-tune ──► best_cnn_model.pth ┘
                                                                        08 fusion config ─┘
```

Shared logic is centralised in `src/` so the notebooks stay thin and consistent:

- **`src/config.py`** — all paths resolved relative to the repo root; raw-dataset locations overridable via environment variables.
- **`src/constants.py`** — the canonical 3-class label names, display colours, metadata-column names, and RNG seed.
- **`src/data_utils.py`** — `load_features()`, which reads `features.csv` and splits it into `X`, `y`, and subject `groups`, dropping the `label`/`subject_id` metadata so `subject_id` can never leak in as a feature.
- **`src/plotting.py`** — a shared confusion-matrix plotting helper used across the training notebooks.

---

## Datasets

### WESAD (Wearable Stress and Affect Detection)
- 15 subjects (S2–S17) wearing chest + wrist physiological sensors.
- Signals used: ECG, EDA, EMG, Respiration (chest device, 700 Hz).
- Labels: Baseline, Stress (TSST protocol), Amusement.
- Windowing: 60-second windows, 30-second step (50% overlap), ≥80% label purity → **1,049 windows** (Baseline 570 / Stress 313 / Amusement 166).
- Download: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

### FER2013 (Facial Expression Recognition)
- 35,887 grayscale face images (48×48).
- 7 emotions mapped to 3 stress states: **Stress** = angry, disgust, fear · **Amusement** = happy, surprise · **Baseline** = neutral, sad.
- Download: https://www.kaggle.com/datasets/msambare/fer2013

Both datasets are large and are **not** committed to the repo (see [Installation](#installation)).

---

## Features Extracted (34 total)

- **HRV / ECG (12):** mean RR, SDNN, RMSSD, pNN50, min/max RR, RR range, mean HR, coefficient of variation, skewness, kurtosis, peak count.
- **EDA (8):** mean, std, min, max, range, skewness, kurtosis, slope.
- **Respiration (8):** mean, std, min, max, range, skewness, kurtosis, slope.
- **EMG (6):** mean absolute, std, max, RMS, skewness, kurtosis.

Constant/zero-variance windows can produce NaN/inf in these statistics; `02_feature_extraction.py` replaces any such values with 0.0 and logs how many windows were affected (on the current dataset: **0 of 1,049**).

---

## Project Structure

```
app/
    app.py                     Streamlit web application (6 pages)

src/
    config.py                  Repo-relative paths + env-var dataset overrides
    constants.py               Label names, colours, metadata columns, RNG seed
    data_utils.py              load_features(): features.csv -> X / y / groups
    plotting.py                Shared confusion-matrix plotting helper
    __init__.py

notebooks/
    01_data_loading.py         WESAD loading + 60s sliding windows (carries subject_id)
    02_feature_extraction.py   34 physiological features + NaN/inf handling
    03_visualization.py        Feature visualizations (5 plots)
    04_model_training.py       SVM / XGBoost / MLP with Leave-One-Subject-Out CV
    05_clustering.py           K-Means, PCA, t-SNE
    06_face_preprocessing.py   FER2013 loading + emotion->stress mapping
    07_train_cnn.py            MobileNetV2 transfer learning
    07b_retrain_cnn.py         MobileNetV2 fine-tuning
    08_fusion.py               Fixed fusion-weight config (no fabricated metric)

models/                        (git-ignored; model binaries hosted on Hugging Face)
    best_model_XGBoost.pkl     Deployed biosignal pipeline (SMOTE + scaler + XGBoost)
    best_model_SVM.pkl         Highest-scoring biosignal pipeline
    best_cnn_model.pth         MobileNetV2 weights
    scaler.pkl / feature_names.pkl / fusion_config.pkl / cnn_config.pkl
    xgb_model_only.pkl / scaler_cloud.pkl   Cloud (SMOTE-free) inference variants

data/processed/
    features.csv               1,049 rows x 36 cols (34 features + label + subject_id)
    plots/                     16 generated PNGs (git-ignored)

Dockerfile / requirements.txt / requirements-hf.txt / runtime.txt / .python-version
```

---

## Installation

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/stress-detection-wesad-ml.git
cd stress-detection-wesad-ml
```

### 2. Virtual environment (Python 3.11)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
`requirements.txt` covers the biosignal pipeline and the app but **not** PyTorch (kept out so CPU-only / cloud installs stay small).

### 4. Install PyTorch (only needed for the CNN / face pipeline)
```bash
# GPU (CUDA 12.1):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# or CPU-only:
pip install torch==2.5.1 torchvision==0.20.1
```

### 5. Datasets
Place the raw datasets under the repo root (the default locations):
```
<repo>/WESAD Dataset/WESAD/      # S2 ... S17 subject folders
<repo>/WESAD Dataset/FER2013/    # train/ and test/ image folders
```
Or point at them anywhere via environment variables:
```bash
export WESAD_DATA_PATH=/path/to/WESAD
export FER2013_DATA_PATH=/path/to/FER2013
```

---

## Running the Pipeline

Scripts build on each other. The **biosignal path** is required for the app's model artifacts:

```bash
python notebooks/01_data_loading.py        # WESAD -> all_windows.npy
python notebooks/02_feature_extraction.py  # -> features.csv
python notebooks/04_model_training.py      # LOSO CV -> models/*.pkl  (must run before the app)
python notebooks/05_clustering.py          # K-Means / PCA / t-SNE plots
python notebooks/03_visualization.py       # feature plots
python notebooks/08_fusion.py              # writes fusion_config.pkl
```

The **face path** (optional, needs PyTorch + FER2013):

```bash
python notebooks/06_face_preprocessing.py  # FER2013 -> face_*.npy
python notebooks/07_train_cnn.py           # initial MobileNetV2 training
python notebooks/07b_retrain_cnn.py        # fine-tuning -> best_cnn_model.pth
```

---

## Running the Streamlit App

```bash
streamlit run app/app.py
```
Then open http://localhost:8501.

### App pages
- **Overview** — pipeline, architecture, dataset stats.
- **Biosignal Predict** — upload an ECG/EDA/EMG/Resp feature CSV, or use sample WESAD data.
- **Face Stress Detection** — upload a face image for the CNN (requires PyTorch).
- **Fusion Predict** — live weighted combination of both models (requires PyTorch for the face side).
- **Model Results** — LOSO metrics, confusion matrices, per-class reports.
- **Visualizations** — browse all 16 generated plots.

The app loads paths from `src/config.py`, so it works from a clean checkout without hardcoded paths. The trained model binaries are git-ignored and hosted separately (Hugging Face); download them into `models/` for full local functionality.

---

## Live Demo

A hosted demo is deployed via Docker (Hugging Face Space frontmatter is at the top of this file). Because PyTorch is heavy for free-tier hosting, the **biosignal prediction, model results, and visualizations** work fully in the cloud, while **face detection and the face side of fusion** are best run locally with PyTorch installed.

> Note: if a public demo URL is present in the project page and it is unreachable, the app can always be run locally with the steps above.

---

## ML Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Signal preprocessing | Butterworth bandpass filter (0.5–40 Hz) on ECG |
| Feature engineering | 34 hand-crafted physiological features |
| Class imbalance | SMOTE oversampling **inside each training fold** |
| Subject-independent evaluation | **Leave-One-Subject-Out** cross-validation |
| Transfer learning | MobileNetV2 pretrained on ImageNet |
| Fine-tuning | Unfreezing the last feature blocks |
| Unsupervised learning | K-Means clustering, PCA, t-SNE |
| Multimodal fusion | Fixed-weight probability combination (live demo) |
| Deployment | Streamlit app, containerised with Docker |

---

## Hardware Used

- CPU: Intel Core i5-12450HX
- GPU: NVIDIA RTX 3050 6GB (CUDA 12.1)
- RAM: 16 GB
- OS: Windows 11

---

## Future Work

- **Deeper subject-independent evaluation** — nested LOSO with per-subject hyperparameter selection, and reporting per-subject confidence intervals rather than a single mean ± std.
- **Closing the LOSO gap** — subject-normalisation / domain-adaptation techniques to reduce the large inter-subject variance, and personalisation from a short calibration window.
- **Genuinely paired multimodal data** — collecting (or sourcing) a dataset with simultaneous biosignals and facial video would, for the first time, make fusion trainable and evaluable against real ground truth.
- **Custom frontend** — a bespoke UI (HTML/CSS/JS + TypeScript, or a React-based app) is being considered as an alternative to Streamlit, for finer control over UX and design, deployed on free-tier hosting (e.g. Vercel for the frontend + Render or Hugging Face for the model API).

---

## References

1. Schmidt, P., et al. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018.
2. Sandler, M., Howard, A., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
3. Goodfellow, I., et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests* (FER2013).

---

## Author

**Anushree** — third-year engineering student. Developed as part of an AI/ML internship in the healthcare domain.

---

## License

Released under the **MIT License**. WESAD and FER2013 are the property of their respective authors and are subject to their own licenses/terms; please cite them as above.
