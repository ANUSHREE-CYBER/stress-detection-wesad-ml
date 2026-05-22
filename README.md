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
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B)

---

## Overview

This project implements an end-to-end **multimodal stress detection pipeline** that classifies human stress into three states — **Baseline**, **Stress**, and **Amusement** — by combining:

- **Biosignal Model:** XGBoost trained on 34 physiological features extracted from ECG, EDA, EMG and Respiration signals (WESAD dataset)
- **Vision Model:** MobileNetV2 CNN trained on facial expression images (FER2013 dataset)
- **Fusion:** Weighted average of both model probability outputs for final prediction

The system is deployed as an interactive **Streamlit web application** with 6 functional pages.

---
## Live Demo

**Try the app here:** https://stress-detection-wesad-ml-tksei97flafzukv2fnrocd.streamlit.app

**What works on the live demo:**
- Overview page with full project pipeline
- Biosignal Predict — upload CSV or use sample WESAD data
- Model Results — detailed metrics, confusion matrices, classification reports
- Visualizations — all 18 generated plots

**What requires local setup (PyTorch too large for cloud):**
- Face Stress Detection via CNN
- Multimodal Fusion prediction

## Architecture

The system uses two parallel models whose probability outputs are combined:

**Model 1 — Biosignal (XGBoost)**
- Input: ECG, EDA, EMG, Respiration signals
- Processing: 34 hand-crafted physiological features
- Accuracy: 97.14%
- Weight in fusion: 60%

**Model 2 — Vision (MobileNetV2 CNN)**
- Input: Face image (48x48 grayscale)
- Processing: Transfer learning from ImageNet
- Accuracy: 63.30%
- Weight in fusion: 40%

**Fusion**
- Method: Weighted average of both model probability outputs
- Final output: Baseline / Stress / Amusement

## Datasets

### WESAD (Wearable Stress and Affect Detection)
- 15 subjects wearing chest + wrist physiological sensors
- Signals: ECG, EDA, EMG, Respiration, Temperature, ACC
- Sampling rate: 700 Hz (chest device)
- Labels: Baseline, Stress (TSST protocol), Amusement
- Download: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

### FER2013 (Facial Expression Recognition)
- 35,887 grayscale face images (48x48 pixels)
- 7 emotion classes mapped to 3 stress states:
  - Stress: angry, disgust, fear
  - Amusement: happy, surprise
  - Baseline: neutral, sad
- Download: https://www.kaggle.com/datasets/msambare/fer2013

---

## Results

| Model | Type | Accuracy | F1 Score |
|---|---|---|---|
| XGBoost | Biosignal | 97.14% | 97.13% |
| SVM | Biosignal | 95.81% | 95.77% |
| MLP | Biosignal | 93.04% | 93.10% |
| MobileNetV2 | Face CNN | 63.30% | 62.00% |

**Evaluation method:** 5-Fold Stratified Cross-Validation + SMOTE for biosignal models. Independent train/test split for CNN.

---

## Features Extracted (34 total)

**HRV Features (12):** Mean RR interval, SDNN, RMSSD, pNN50, Min/Max RR, HRV range, Mean HR, CV, Skewness, Kurtosis, Peak count

**EDA Features (8):** Mean, Std, Min, Max, Range, Skewness, Kurtosis, Slope

**Respiration Features (8):** Mean, Std, Min, Max, Range, Skewness, Kurtosis, Slope

**EMG Features (6):** Mean absolute, Std, Max, RMS, Skewness, Kurtosis

---


## Project Structure

    app/
        app.py                     Streamlit web application (6 pages)

    notebooks/
        01_data_loading.py         WESAD data loading and sliding windows
        02_feature_extraction.py   34 physiological feature extraction
        03_visualization.py        Data visualizations (10 plots)
        04_model_training.py       SVM, XGBoost, MLP training
        05_clustering.py           K-Means, PCA, t-SNE analysis
        06_face_preprocessing.py   FER2013 loading and label mapping
        07_train_cnn.py            MobileNetV2 initial training
        07b_retrain_cnn.py         MobileNetV2 fine-tuning
        08_fusion.py               Multimodal fusion analysis

    models/
        best_model_XGBoost.pkl     Trained XGBoost pipeline
        best_cnn_model.pth         Trained CNN weights
        fusion_config.pkl          Fusion weights and config
        scaler.pkl                 Feature scaler
        feature_names.pkl          List of 34 feature names

    data/processed/
        features.csv               1049 x 34 extracted features
        plots/                     18 saved visualization PNGs

    requirements.txt
    .gitignore
    README.md

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stress-detection-wesad-ml.git
cd stress-detection-wesad-ml
```

### 2. Create virtual environment (Python 3.11 recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA (for GPU support)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Download datasets
- Place WESAD dataset in: `D:/WESAD Dataset/WESAD/`
- Place FER2013 dataset in: `D:/WESAD Dataset/FER2013/`

---

## Running the Pipeline

Run scripts in order — each builds on the previous:

```bash
# Step 1: Load and window WESAD data
python notebooks/01_data_loading.py

# Step 2: Extract physiological features
python notebooks/02_feature_extraction.py

# Step 3: Generate visualizations
python notebooks/03_visualization.py

# Step 4: Train biosignal ML models
python notebooks/04_model_training.py

# Step 5: Unsupervised clustering analysis
python notebooks/05_clustering.py

# Step 6: Preprocess FER2013 face images
python notebooks/06_face_preprocessing.py

# Step 7: Train CNN
python notebooks/07_train_cnn.py

# Step 8: Fine-tune CNN
python notebooks/07b_retrain_cnn.py

# Step 9: Fusion analysis
python notebooks/08_fusion.py
```

---

## Running the Streamlit App

```bash
streamlit run app/app.py
```

Open browser at: http://localhost:8501

### App Pages:
- **Overview** — Project pipeline, architecture, dataset stats
- **Biosignal Predict** — Upload ECG/EDA features CSV for XGBoost prediction
- **Face Stress Detection** — Upload face image for CNN prediction
- **Fusion Predict** — Combined biosignal + face prediction
- **Model Results** — Detailed metrics, confusion matrices, reports
- **Visualizations** — Browse all 18 generated plots

---

## ML Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Signal preprocessing | Butterworth bandpass filter (0.5-40Hz) |
| Feature engineering | 34 hand-crafted physiological features |
| Class imbalance | SMOTE oversampling inside each CV fold |
| Model evaluation | 5-Fold Stratified Cross-Validation |
| Transfer learning | MobileNetV2 pretrained on ImageNet |
| Fine-tuning | Unfreezing last 5 feature blocks |
| Unsupervised learning | K-Means clustering, PCA, t-SNE |
| Multimodal fusion | Weighted probability averaging |
| Deployment | Streamlit web application |

---

## Hardware Used

- CPU: Intel Core i5-12450HX (8 cores)
- GPU: NVIDIA RTX 3050 6GB (CUDA 12.1)
- RAM: 16GB
- OS: Windows 11

---

## References

1. Schmidt, P., et al. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. ICMI 2018.
2. Howard, A., et al. (2019). Searching for MobileNetV2. CVPR 2018.
3. Goodfellow, I., et al. (2013). Challenges in representation learning: A report on three machine learning contests. FER2013.

---

## Author

**Anushree**
Third Year Engineering Student
Project developed as part of AI/ML internship in Healthcare domain.

---

## License

This project is licensed under the MIT License.
