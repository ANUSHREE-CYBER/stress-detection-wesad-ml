import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────
TRAIN_PATH     = r"D:\WESAD Dataset\FER2013\train"
TEST_PATH      = r"D:\WESAD Dataset\FER2013\test"
OUTPUT_PATH    = r"D:\STRESS DETECTION\data\processed"
PLOTS_PATH     = r"D:\STRESS DETECTION\data\processed\plots"
IMG_SIZE       = 48

# Emotion → Stress label mapping
# 0=Baseline, 1=Stress, 2=Amusement  (same as WESAD)
EMOTION_MAP = {
    'angry'   : 1,   # Stress
    'disgust' : 1,   # Stress
    'fear'    : 1,   # Stress
    'happy'   : 2,   # Amusement
    'neutral' : 0,   # Baseline
    'sad'     : 0,   # Baseline
    'surprise': 2,   # Amusement
}
LABEL_NAMES = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}

# ── LOAD IMAGES ───────────────────────────────────────────────────────────
def load_images(base_path, split_name):
    images = []
    labels = []
    counts = defaultdict(int)

    print(f"\nLoading {split_name} images...")
    for emotion, label in EMOTION_MAP.items():
        folder = os.path.join(base_path, emotion)
        if not os.path.exists(folder):
            print(f"  WARNING: {folder} not found, skipping")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith('.jpg')]

        for fname in files:
            img_path = os.path.join(folder, fname)
            try:
                img = Image.open(img_path).convert('L')  # grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(label)
                counts[emotion] += 1
            except Exception as e:
                print(f"  Error loading {fname}: {e}")

        print(f"  {emotion:10s} → {LABEL_NAMES[label]:10s}: "
              f"{counts[emotion]} images")

    return np.array(images), np.array(labels), counts

# ── LOAD TRAIN AND TEST ───────────────────────────────────────────────────
X_train, y_train, train_counts = load_images(TRAIN_PATH, "TRAIN")
X_test,  y_test,  test_counts  = load_images(TEST_PATH,  "TEST")

print(f"\nX_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_test  shape : {y_test.shape}")

# ── LABEL DISTRIBUTION ────────────────────────────────────────────────────
print("\nTrain label distribution:")
for lbl, name in LABEL_NAMES.items():
    count = int(np.sum(y_train == lbl))
    pct   = count / len(y_train) * 100
    print(f"  {name:12s}: {count:6d} ({pct:.1f}%)")

print("\nTest label distribution:")
for lbl, name in LABEL_NAMES.items():
    count = int(np.sum(y_test == lbl))
    pct   = count / len(y_test) * 100
    print(f"  {name:12s}: {count:6d} ({pct:.1f}%)")

# ── SAVE PROCESSED ARRAYS ────────────────────────────────────────────────
print("\nSaving processed arrays...")
np.save(os.path.join(OUTPUT_PATH, "face_X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "face_y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "face_X_test.npy"),  X_test)
np.save(os.path.join(OUTPUT_PATH, "face_y_test.npy"),  y_test)
print(f"Saved to {OUTPUT_PATH}")

# ── PLOT: Sample images per class ────────────────────────────────────────
print("\nGenerating sample image plot...")
fig, axes = plt.subplots(3, 5, figsize=(14, 9))
COLORS = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}

for row, (lbl, name) in enumerate(LABEL_NAMES.items()):
    indices = np.where(y_train == lbl)[0][:5]
    for col, idx in enumerate(indices):
        axes[row, col].imshow(X_train[idx], cmap='gray')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(name, fontsize=12,
                                       fontweight='bold',
                                       color=COLORS[lbl])
            axes[row, col].axis('on')
            axes[row, col].set_yticks([])
            axes[row, col].set_xticks([])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(COLORS[lbl])
                spine.set_linewidth(3)

fig.suptitle('Sample Face Images per Stress Class\n(48×48 grayscale)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "11_sample_faces.png"), dpi=150)
plt.close()
print("  Saved: 11_sample_faces.png")

# ── PLOT: Class distribution comparison ──────────────────────────────────
print("Generating class distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
label_list  = list(LABEL_NAMES.values())
colors_list = [COLORS[i] for i in range(3)]

for ax, (split_y, title) in zip(axes, [
        (y_train, 'Train Set'),
        (y_test,  'Test Set')]):
    counts_arr = [int(np.sum(split_y == i)) for i in range(3)]
    bars = ax.bar(label_list, counts_arr,
                  color=colors_list, edgecolor='black')
    for bar, val in zip(bars, counts_arr):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 50,
                str(val), ha='center', fontweight='bold')
    ax.set_title(f'FER2013 {title} — Stress Label Distribution',
                 fontweight='bold')
    ax.set_ylabel('Number of Images')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "12_face_distribution.png"), dpi=150)
plt.close()
print("  Saved: 12_face_distribution.png")

print("\n✅ Face preprocessing complete!")
print(f"Memory used — X_train: "
      f"{X_train.nbytes/1024**2:.1f} MB, "
      f"X_test: {X_test.nbytes/1024**2:.1f} MB")