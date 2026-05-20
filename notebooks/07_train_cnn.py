import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────
PROCESSED_PATH = r"D:\STRESS DETECTION\data\processed"
MODELS_PATH    = r"D:\STRESS DETECTION\models"
PLOTS_PATH     = r"D:\STRESS DETECTION\data\processed\plots"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 64
EPOCHS         = 20
LR             = 0.001
NUM_CLASSES    = 3
RANDOM_STATE   = 42

torch.manual_seed(RANDOM_STATE)
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── CUSTOM DATASET ────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # Convert grayscale to 3-channel RGB for MobileNetV2
        # images shape: (N, 48, 48) → (N, 3, 48, 48)
        self.images    = torch.tensor(images, dtype=torch.float32)
        self.images    = self.images.unsqueeze(1).repeat(1, 3, 1, 1)
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ── DATA AUGMENTATION ─────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── LOAD DATA ─────────────────────────────────────────────────────────────
print("\nLoading preprocessed face arrays...")
X_train = np.load(os.path.join(PROCESSED_PATH, "face_X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_PATH, "face_y_train.npy"))
X_test  = np.load(os.path.join(PROCESSED_PATH, "face_X_test.npy"))
y_test  = np.load(os.path.join(PROCESSED_PATH, "face_y_test.npy"))

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

train_dataset = FaceDataset(X_train, y_train, transform=train_transform)
test_dataset  = FaceDataset(X_test,  y_test,  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ── BUILD MODEL (MobileNetV2 Transfer Learning) ───────────────────────────
print("\nBuilding MobileNetV2 model...")
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

# Freeze ALL layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 2 feature layers + classifier
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Replace classifier for 3 classes
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(128, NUM_CLASSES)
)

model = model.to(DEVICE)

# Count trainable parameters
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,} "
      f"({trainable_params/total_params*100:.1f}%)")

# ── TRAINING SETUP ────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs on {DEVICE}...")
print("-" * 60)

train_losses, test_losses   = [], []
train_accs,   test_accs     = [], []
best_acc = 0.0

for epoch in range(EPOCHS):
    # ── TRAIN ──
    model.train()
    running_loss    = 0.0
    correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = correct / total * 100

    # ── EVALUATE ──
    model.eval()
    running_loss    = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc  = correct / total * 100

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    scheduler.step()

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
          f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),
                   os.path.join(MODELS_PATH, "best_cnn_model.pth"))
        print(f"  ✅ New best model saved! ({best_acc:.2f}%)")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")

# ── SAVE FINAL MODEL ──────────────────────────────────────────────────────
torch.save(model.state_dict(),
           os.path.join(MODELS_PATH, "final_cnn_model.pth"))
joblib.dump({'num_classes': NUM_CLASSES,
             'img_size': 48,
             'best_acc': best_acc},
            os.path.join(MODELS_PATH, "cnn_config.pkl"))
print("Saved: final_cnn_model.pth, cnn_config.pkl")

# ── PLOT: Training curves ─────────────────────────────────────────────────
print("\nGenerating training curve plots...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
epochs_range = range(1, EPOCHS + 1)

axes[0].plot(epochs_range, train_losses, 'b-o', markersize=4, label='Train Loss')
axes[0].plot(epochs_range, test_losses,  'r-o', markersize=4, label='Test Loss')
axes[0].set_title('Training & Validation Loss', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, train_accs, 'b-o', markersize=4, label='Train Acc')
axes[1].plot(epochs_range, test_accs,  'r-o', markersize=4, label='Test Acc')
axes[1].set_title('Training & Validation Accuracy', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 100)

fig.suptitle(f'MobileNetV2 Training Curves  '
             f'(Best Test Acc: {best_acc:.2f}%)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "13_cnn_training_curves.png"), dpi=150)
plt.close()
print("  Saved: 13_cnn_training_curves.png")

# ── PLOT: Confusion matrix ────────────────────────────────────────────────
print("Generating CNN confusion matrix...")
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

LABEL_NAMES = ['Baseline', 'Stress', 'Amusement']
cm = confusion_matrix(all_labels, all_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES, ax=ax)
ax.set_title(f'CNN Confusion Matrix  (Test Acc: {best_acc:.2f}%)',
             fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "14_cnn_confusion_matrix.png"), dpi=150)
plt.close()
print("  Saved: 14_cnn_confusion_matrix.png")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
      target_names=LABEL_NAMES))
print("\n✅ CNN Training complete!")