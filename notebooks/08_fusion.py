"""Fusion configuration for multimodal stress detection.

WHY THESE WEIGHTS ARE FIXED (NOT LEARNED)
-----------------------------------------
The biosignal model (XGBoost, trained on WESAD) and the facial model
(MobileNetV2, trained on FER2013) come from two datasets that share ZERO
subjects. There is no recording anywhere of the same person's biosignals AND
facial expression captured at the same moment under a common stress label.

That means there is no paired ground-truth dataset on which a fusion rule
could be either TRAINED or EVALUATED. Any reported "fusion accuracy" would
therefore be fabricated — the only way to produce one is to invent paired
data or to score a combined prediction against labels it was built from
(which is circular).

The previous version of this file did exactly that: it wrote 0.97 onto each
sample's TRUE label to synthesise a "confident XGBoost", fused that with the
CNN probabilities, then "measured" accuracy against those same true labels.
That number was meaningless and has been removed entirely.

WHAT THIS FILE DOES INSTEAD
---------------------------
It defines FIXED, heuristic fusion weights and saves them to
``fusion_config.pkl`` (the file the Streamlit app already loads). Fusion is a
LIVE app feature: when a user supplies both a biosignal reading and a face
image, the app runs each model independently and combines their class
probabilities with these fixed weights. It is a proof-of-concept for
multimodal deployment (e.g. a wearable + webcam), NOT a benchmarked model,
and no accuracy is claimed for it anywhere.

WEIGHT CHOICE: 0.6 biosignal / 0.4 facial
-----------------------------------------
Biosignals (ECG/EDA/EMG/respiration) are direct correlates of the autonomic
stress response, and the biosignal model is the stronger of the two under
honest subject-wise (Leave-One-Subject-Out) evaluation — roughly 74.6% mean
per-subject accuracy vs ~63.3% for the face CNN. Facial expression is a
noisier proxy for stress (it conflates general emotion with stress and is
easily confounded by lighting, pose, and identity). We therefore weight the
biosignal modality higher. These weights are a reasonable prior, NOT an
optimised value — adjust them if real deployment experience suggests a
different balance.
"""
import os
import sys
import joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR

MODELS_PATH = MODELS_DIR
LABEL_NAMES = ['Baseline', 'Stress', 'Amusement']

# ── FIXED FUSION WEIGHTS ──────────────────────────────────────────────────
XGB_WEIGHT = 0.6   # biosignal (XGBoost)  — stronger + more direct stress signal
CNN_WEIGHT = 0.4   # facial   (MobileNetV2) — noisier proxy
assert abs(XGB_WEIGHT + CNN_WEIGHT - 1.0) < 1e-9, "fusion weights must sum to 1.0"

fusion_config = {
    'xgb_weight': XGB_WEIGHT,
    'cnn_weight': CNN_WEIGHT,
    'method':     'fixed_heuristic',
    'note':       ('Fixed heuristic weights — not learned and not evaluated. '
                   'WESAD and FER2013 share no subjects, so no paired ground '
                   'truth exists and no real fusion accuracy can be computed.'),
}

os.makedirs(MODELS_PATH, exist_ok=True)
joblib.dump(fusion_config, os.path.join(MODELS_PATH, "fusion_config.pkl"))
print("Saved fusion_config.pkl:")
for k, v in fusion_config.items():
    print(f"  {k}: {v}")

# ── ILLUSTRATIVE DEMO (NOT AN EVALUATION) ─────────────────────────────────
# The pairs below are HAND-MADE examples used only to show HOW the weighted
# combination behaves. They are NOT real data, involve NO ground-truth labels,
# and produce NO accuracy metric. This mirrors the exact arithmetic the app
# performs live on a user's biosignal + face inputs.
def fuse(bio_probs, face_probs):
    bio  = np.asarray(bio_probs,  dtype=float)
    face = np.asarray(face_probs, dtype=float)
    return XGB_WEIGHT * bio + CNN_WEIGHT * face

print("\n=== ILLUSTRATIVE FUSION EXAMPLES "
      "(made-up inputs — NOT an accuracy test) ===")
examples = [
    ("biosignal leans Stress, face looks Neutral",
     [0.10, 0.80, 0.10], [0.50, 0.25, 0.25]),
    ("biosignal leans Baseline, face looks Amused",
     [0.75, 0.15, 0.10], [0.20, 0.10, 0.70]),
    ("conflicting: biosignal Stress vs face Amused",
     [0.15, 0.70, 0.15], [0.15, 0.10, 0.75]),
]
for desc, bio, face in examples:
    fused = fuse(bio, face)
    pred  = LABEL_NAMES[int(np.argmax(fused))]
    print(f"\n  {desc}")
    print(f"    biosignal probs      : {np.round(bio,   2)}")
    print(f"    facial probs         : {np.round(face,  2)}")
    print(f"    fused (0.6/0.4)      : {np.round(fused, 2)}  ->  {pred}")

print("\nFusion config written. No training, no evaluation — by design.")
