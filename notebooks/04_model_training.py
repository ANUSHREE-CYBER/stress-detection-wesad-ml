import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, MODELS_DIR, PLOTS_DIR
from src.constants import LABEL_NAMES_LIST, RANDOM_STATE
from src.data_utils import load_features
from src.plotting import plot_confusion_matrix

# ── CONFIG ───────────────────────────────────────────────────────────────
PROCESSED_PATH = DATA_DIR
MODELS_PATH    = MODELS_DIR
PLOTS_PATH     = PLOTS_DIR
os.makedirs(MODELS_PATH, exist_ok=True)

# ── LOAD FEATURES ─────────────────────────────────────────────────────────
print("Loading features...")
X, y, groups, feature_names = load_features(PROCESSED_PATH)
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Subjects: {np.unique(groups)}")
print(f"Classes: {np.unique(y, return_counts=True)}")

# ── CROSS VALIDATION SETUP (subject-aware) ────────────────────────────────
# Leave-One-Subject-Out: train on N-1 subjects, test on the held-out subject.
# This prevents windows from the same subject landing in both train and test,
# which inflated accuracy under the old random StratifiedKFold split.
cv = LeaveOneGroupOut()

# ── DEFINE MODELS ─────────────────────────────────────────────────────────
models = {
    'SVM': ImbPipeline([
        ('smote',   SMOTE(random_state=RANDOM_STATE)),
        ('scaler',  StandardScaler()),
        ('model',   SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=RANDOM_STATE))
    ]),
    'XGBoost': ImbPipeline([
        ('smote',   SMOTE(random_state=RANDOM_STATE)),
        ('scaler',  StandardScaler()),
        ('model',   XGBClassifier(n_estimators=200, max_depth=6,
                                  learning_rate=0.1, subsample=0.8,
                                  eval_metric='mlogloss',
                                  random_state=RANDOM_STATE,
                                  verbosity=0))
    ]),
    'MLP': ImbPipeline([
        ('smote',   SMOTE(random_state=RANDOM_STATE)),
        ('scaler',  StandardScaler()),
        ('model',   MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                  activation='relu', max_iter=500,
                                  early_stopping=True, validation_fraction=0.1,
                                  random_state=RANDOM_STATE))
    ])
}

# ── TRAIN & EVALUATE (Leave-One-Subject-Out CV) ──────────────────────────
results = {}
subjects = sorted(np.unique(groups), key=lambda s: int(str(s)[1:]))

for model_name, pipeline in models.items():
    print(f"\nTraining {model_name} (Leave-One-Subject-Out, {len(subjects)} folds)...")

    # Out-of-fold predictions: each window is predicted only when ITS subject
    # is the held-out test fold. SMOTE runs inside each training fold only.
    y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=cv, n_jobs=-1)

    # Per-subject accuracy = accuracy on the windows of the held-out subject
    per_subject_acc = {s: accuracy_score(y[groups == s], y_pred[groups == s])
                       for s in subjects}
    subj_accs = np.array([per_subject_acc[s] for s in subjects])
    mean_acc  = subj_accs.mean()
    std_acc   = subj_accs.std()
    f1        = f1_score(y, y_pred, average='weighted')

    results[model_name] = {
        'accuracy':        mean_acc,   # mean per-subject accuracy — replaces old single-CV accuracy
        'accuracy_std':    std_acc,
        'per_subject_acc': per_subject_acc,
        'f1_score':        f1,
        'y_pred':          y_pred,
    }

    print(f"  Mean subject accuracy : {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (std across subjects)")
    print(f"  Weighted F1 (pooled)  : {f1*100:.2f}%")
    print(f"  Per-subject accuracy:")
    for s in subjects:
        print(f"    {str(s):>4}: {per_subject_acc[s]*100:5.1f}%")
    print(f"\n  Classification Report (pooled out-of-fold predictions):")
    print(classification_report(y, y_pred,
          target_names=LABEL_NAMES_LIST))

# ── SAVE BEST MODEL ───────────────────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]['f1_score'])
print(f"\nBest model: {best_model_name} "
      f"(F1={results[best_model_name]['f1_score']*100:.2f}%)")

# Retrain best model on full data and save
print(f"Retraining {best_model_name} on full dataset...")
best_pipeline = models[best_model_name]
best_pipeline.fit(X, y)
joblib.dump(best_pipeline,
            os.path.join(MODELS_PATH, f"best_model_{best_model_name}.pkl"))
print(f"Saved to models/best_model_{best_model_name}.pkl")

# ── ALSO SAVE XGBoost AS THE DEPLOYED MODEL ───────────────────────────────
# The comparison above stays honest (SVM may score higher), but XGBoost is the
# model we actually deploy in app.py / fusion (faster inference, accuracy within
# std of the best). Retrain it on the full dataset and save unconditionally so a
# fresh LOSO-pipeline artifact always exists, regardless of which model "won".
if best_model_name != 'XGBoost':
    print("Retraining XGBoost on full dataset (deployed model)...")
    xgb_pipeline = models['XGBoost']
    xgb_pipeline.fit(X, y)
    joblib.dump(xgb_pipeline,
                os.path.join(MODELS_PATH, "best_model_XGBoost.pkl"))
    print("Saved to models/best_model_XGBoost.pkl")

# Also save scaler separately for frontend use
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

# Save feature names (metadata columns excluded — model sees features only)
joblib.dump(feature_names, os.path.join(MODELS_PATH, "feature_names.pkl"))
print("Saved scaler and feature names")

# ── PLOT: Confusion Matrices ───────────────────────────────────────────────
print("\nGenerating confusion matrix plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (model_name, res) in zip(axes, results.items()):
    plot_confusion_matrix(ax, y, res['y_pred'], LABEL_NAMES_LIST, cbar=False)
    ax.set_title(f'{model_name}\nAcc={res["accuracy"]*100:.1f}±{res["accuracy_std"]*100:.1f}%  '
                 f'F1={res["f1_score"]*100:.1f}%',
                 fontweight='bold')

fig.suptitle('Confusion Matrices (Leave-One-Subject-Out CV) — % of Actual',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "06_confusion_matrices.png"), dpi=150)
plt.close()
print("  Saved: 06_confusion_matrices.png")

# ── PLOT: Model Comparison Bar Chart ──────────────────────────────────────
print("Generating model comparison plot...")
fig, ax = plt.subplots(figsize=(9, 5))
model_names = list(results.keys())
accs      = [results[m]['accuracy']*100     for m in model_names]
acc_stds  = [results[m]['accuracy_std']*100 for m in model_names]
f1s       = [results[m]['f1_score']*100     for m in model_names]
x    = np.arange(len(model_names))
w    = 0.35

# Accuracy bars carry error bars showing per-subject std (LOSO variance)
bars1 = ax.bar(x - w/2, accs, w, label='Mean Subject Accuracy',
               color='#42A5F5', edgecolor='black',
               yerr=acc_stds, capsize=4)
bars2 = ax.bar(x + w/2, f1s,  w, label='F1 Score',
               color='#EF5350', edgecolor='black')

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%',
            ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 110)
ax.set_ylabel('Score (%)')
ax.set_title('Model Comparison — Accuracy & F1 Score\n(Leave-One-Subject-Out CV + SMOTE)',
             fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "07_model_comparison.png"), dpi=150)
plt.close()
print("  Saved: 07_model_comparison.png")

print("\n✅ Model training complete!")
print(f"Best model: {best_model_name}")
print(f"Saved in: {MODELS_PATH}")
