import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

PROCESSED_PATH = r"D:\STRESS DETECTION\data\processed"
PLOTS_PATH     = r"D:\STRESS DETECTION\data\processed\plots"
os.makedirs(PLOTS_PATH, exist_ok=True)

LABEL_NAMES = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}
COLORS      = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}

print("Loading features...")
df      = pd.read_csv(os.path.join(PROCESSED_PATH, "features.csv"))
X       = df.drop('label', axis=1).values
y       = df['label'].values
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Shape: {X_scaled.shape}")

# ── ELBOW METHOD ──────────────────────────────────────────────────────────
print("Running elbow method...")
inertias    = []
sil_scores  = []
k_range     = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={sil_scores[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3 (optimal)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), sil_scores, 'ro-', linewidth=2, markersize=8)
axes[1].axvline(x=3, color='blue', linestyle='--', alpha=0.7, label='k=3')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs k', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('K-Means Cluster Selection', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "08_elbow_method.png"), dpi=150)
plt.close()
print("  Saved: 08_elbow_method.png")

# ── K-MEANS WITH k=3 ─────────────────────────────────────────────────────
print("\nRunning K-Means with k=3...")
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = km3.fit_predict(X_scaled)

# ── PCA VISUALIZATION ─────────────────────────────────────────────────────
print("Running PCA...")
pca    = PCA(n_components=2, random_state=42)
X_pca  = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA coloured by true labels
for lbl, name in LABEL_NAMES.items():
    mask = y == lbl
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=COLORS[lbl], label=name, alpha=0.6, s=30)
axes[0].set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)')
axes[0].set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)')
axes[0].set_title('PCA — True Labels', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# PCA coloured by cluster
cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for c in range(3):
    mask = cluster_labels == c
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=cluster_colors[c], label=f'Cluster {c}',
                    alpha=0.6, s=30)
axes[1].set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)')
axes[1].set_title('PCA — K-Means Clusters (k=3)', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('PCA Projection of Physiological Features', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "09_pca_visualization.png"), dpi=150)
plt.close()
print("  Saved: 09_pca_visualization.png")

# ── t-SNE VISUALIZATION ───────────────────────────────────────────────────
print("Running t-SNE (this takes ~1-2 minutes)...")
tsne   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for lbl, name in LABEL_NAMES.items():
    mask = y == lbl
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=COLORS[lbl], label=name, alpha=0.6, s=30)
axes[0].set_title('t-SNE — True Labels', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

for c in range(3):
    mask = cluster_labels == c
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=cluster_colors[c], label=f'Cluster {c}',
                    alpha=0.6, s=30)
axes[1].set_title('t-SNE — K-Means Clusters', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.suptitle('t-SNE Projection of Physiological Features', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "10_tsne_visualization.png"), dpi=150)
plt.close()
print("  Saved: 10_tsne_visualization.png")

print(f"\nFinal silhouette score (k=3): {silhouette_score(X_scaled, cluster_labels):.3f}")
print("\n✅ Clustering complete!")