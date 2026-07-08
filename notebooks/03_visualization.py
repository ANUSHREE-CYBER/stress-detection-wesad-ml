import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

print("Starting visualization...")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, PLOTS_DIR

PROCESSED_PATH = DATA_DIR
OUTPUT_PATH    = PLOTS_DIR
os.makedirs(OUTPUT_PATH, exist_ok=True)

LABEL_NAMES = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}
COLORS      = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}

print("Loading features CSV...")
df = pd.read_csv(os.path.join(PROCESSED_PATH, "features.csv"))
print(f"Loaded: {df.shape}")

# PLOT 1: Label Distribution
print("Creating Plot 1: Label distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
counts = df['label'].value_counts().sort_index()
bars = ax.bar(
    [LABEL_NAMES[i] for i in counts.index],
    counts.values,
    color=[COLORS[i] for i in counts.index],
    edgecolor='black'
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            str(val), ha='center', fontweight='bold')
ax.set_title('Class Distribution of Stress Labels', fontweight='bold')
ax.set_ylabel('Number of Windows')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "01_label_distribution.png"), dpi=150)
plt.close()
print("  Saved plot 1")

# PLOT 2: Boxplots of key features
print("Creating Plot 2: Feature boxplots...")
key_features = ['hrv_mean_hr', 'hrv_rmssd', 'hrv_pnn50',
                'eda_mean', 'eda_std', 'resp_std']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    data_by_class = [df[df['label']==lbl][feat].values for lbl in [0,1,2]]
    bp = axes[i].boxplot(data_by_class,
                         labels=list(LABEL_NAMES.values()),
                         patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, lbl in zip(bp['boxes'], [0,1,2]):
        patch.set_facecolor(COLORS[lbl])
        patch.set_alpha(0.7)
    axes[i].set_title(feat.replace('_',' ').title(), fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)
fig.suptitle('Key Feature Distributions by Stress Class', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "02_feature_boxplots.png"), dpi=150)
plt.close()
print("  Saved plot 2")

# PLOT 3: Correlation Heatmap
print("Creating Plot 3: Correlation heatmap...")
fig, ax = plt.subplots(figsize=(14, 11))
corr = df.drop(['label', 'subject_id'], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm',
            center=0, square=True, linewidths=0.3, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "03_correlation_heatmap.png"), dpi=150)
plt.close()
print("  Saved plot 3")

# PLOT 4: Violin plots
print("Creating Plot 4: Violin plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, feat in zip(axes, ['hrv_mean_hr', 'hrv_rmssd']):
    parts = ax.violinplot(
        [df[df['label']==lbl][feat].values for lbl in [0,1,2]],
        positions=[0,1,2], showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.7)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(list(LABEL_NAMES.values()))
    ax.set_title(feat.replace('_',' ').title())
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Heart Rate and RMSSD by Stress Class', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "04_violin_plots.png"), dpi=150)
plt.close()
print("  Saved plot 4")

# PLOT 5: Feature means per class (bar chart)
print("Creating Plot 5: Feature means per class...")
fig, ax = plt.subplots(figsize=(12, 5))
features_to_compare = ['hrv_mean_hr','hrv_rmssd','eda_mean','eda_std','resp_std','emg_rms']
x = np.arange(len(features_to_compare))
width = 0.25
for i, (lbl, name) in enumerate(LABEL_NAMES.items()):
    means = [df[df['label']==lbl][f].mean() for f in features_to_compare]
    means_norm = [m / df[f].mean() for m, f in zip(means, features_to_compare)]
    ax.bar(x + i*width, means_norm, width, label=name,
           color=COLORS[lbl], edgecolor='black', alpha=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels([f.replace('_','\n') for f in features_to_compare], fontsize=8)
ax.set_ylabel('Normalized Mean (relative to overall mean)')
ax.set_title('Normalized Feature Means per Stress Class', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "05_feature_means.png"), dpi=150)
plt.close()
print("  Saved plot 5")

print("\nAll plots saved to:", OUTPUT_PATH)
print("Done!")
