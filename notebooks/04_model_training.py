import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── CONFIG ───────────────────────────────────────────────────────────────
PROCESSED_PATH = r"D:\STRESS DETECTION\data\processed"
MODELS_PATH    = r"D:\STRESS DETECTION\models"
PLOTS_PATH     = r"D:\STRESS DETECTION\data\processed\plots"
os.makedirs(MODELS_PATH, exist_ok=True)

LABEL_NAMES = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}
COLORS      = ['#2196F3', '#F44336', '#4CAF50']
RANDOM_STATE = 42

# ── LOAD FEATURES ─────────────────────────────────────────────────────────
print("Loading features...")
df = pd.read_csv(os.path.join(PROCESSED_PATH, "features.csv"))
X  = df.drop('label', axis=1).values
y  = df['label'].values
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Classes: {np.unique(y, return_counts=True)}")

# ── CROSS VALIDATION SETUP ────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

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

# ── TRAIN & EVALUATE ──────────────────────────────────────────────────────
results = {}

for model_name, pipeline in models.items():
    print(f"\nTraining {model_name}...")

    # Cross-validated predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average='weighted')

    results[model_name] = {
        'accuracy': acc,
        'f1_score': f1,
        'y_pred':   y_pred
    }

    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred,
          target_names=list(LABEL_NAMES.values())))

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

# Also save scaler separately for frontend use
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

# Save feature names
feature_names = df.drop('label', axis=1).columns.tolist()
joblib.dump(feature_names, os.path.join(MODELS_PATH, "feature_names.pkl"))
print("Saved scaler and feature names")

# ── PLOT: Confusion Matrices ───────────────────────────────────────────────
print("\nGenerating confusion matrix plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (model_name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y, res['y_pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=list(LABEL_NAMES.values()),
                yticklabels=list(LABEL_NAMES.values()),
                ax=ax, cbar=False)
    ax.set_title(f'{model_name}\nAcc={res["accuracy"]*100:.1f}%  '
                 f'F1={res["f1_score"]*100:.1f}%',
                 fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

fig.suptitle('Confusion Matrices (5-Fold Cross Validation) — % of Actual',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "06_confusion_matrices.png"), dpi=150)
plt.close()
print("  Saved: 06_confusion_matrices.png")

# ── PLOT: Model Comparison Bar Chart ──────────────────────────────────────
print("Generating model comparison plot...")
fig, ax = plt.subplots(figsize=(9, 5))
model_names = list(results.keys())
accs = [results[m]['accuracy']*100 for m in model_names]
f1s  = [results[m]['f1_score']*100  for m in model_names]
x    = np.arange(len(model_names))
w    = 0.35

bars1 = ax.bar(x - w/2, accs, w, label='Accuracy',
               color='#42A5F5', edgecolor='black')
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
ax.set_title('Model Comparison — Accuracy & F1 Score\n(5-Fold Stratified CV + SMOTE)',
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
