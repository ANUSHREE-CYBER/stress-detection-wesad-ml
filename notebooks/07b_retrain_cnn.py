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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────
PROCESSED_PATH = r"D:\STRESS DETECTION\data\processed"
MODELS_PATH    = r"D:\STRESS DETECTION\models"
PLOTS_PATH     = r"D:\STRESS DETECTION\data\processed\plots"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 64
EPOCHS         = 15
LR             = 0.0003   # lower LR for fine-tuning
NUM_CLASSES    = 3
print(f"Using device: {DEVICE}")

# ── DATASET ───────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = torch.tensor(images, dtype=torch.float32)
        self.images    = self.images.unsqueeze(1).repeat(1, 3, 1, 1)
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── LOAD DATA ─────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load(os.path.join(PROCESSED_PATH, "face_X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_PATH, "face_y_train.npy"))
X_test  = np.load(os.path.join(PROCESSED_PATH, "face_X_test.npy"))
y_test  = np.load(os.path.join(PROCESSED_PATH, "face_y_test.npy"))

train_loader = DataLoader(FaceDataset(X_train, y_train, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(FaceDataset(X_test, y_test, test_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── LOAD PREVIOUS BEST MODEL ──────────────────────────────────────────────
print("Loading best model from previous training...")
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(128, NUM_CLASSES)
)
model.load_state_dict(torch.load(
    os.path.join(MODELS_PATH, "best_cnn_model.pth"),
    map_location=DEVICE))
model = model.to(DEVICE)
print("Model loaded successfully")

# ── UNFREEZE LAST 5 FEATURE BLOCKS ───────────────────────────────────────
print("Unfreezing last 5 feature blocks...")
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 5 blocks + classifier
for param in model.features[-5:].parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# ── TRAINING ──────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\nFine-tuning for {EPOCHS} epochs...")
print("-" * 60)

train_losses, test_losses = [], []
train_accs,   test_accs   = [], []
best_acc = 56.07  # start from previous best

for epoch in range(EPOCHS):
    # Train
    model.train()
    r_loss = correct = total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        r_loss  += loss.item()
        _, pred  = torch.max(out, 1)
        total   += lbls.size(0)
        correct += (pred == lbls).sum().item()
    train_loss = r_loss / len(train_loader)
    train_acc  = correct / total * 100

    # Evaluate
    model.eval()
    r_loss = correct = total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, lbls)
            r_loss  += loss.item()
            _, pred  = torch.max(out, 1)
            total   += lbls.size(0)
            correct += (pred == lbls).sum().item()
    test_loss = r_loss / len(test_loader)
    test_acc  = correct / total * 100

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    scheduler.step()

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
          f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),
                   os.path.join(MODELS_PATH, "best_cnn_model.pth"))
        print(f"  ✅ New best saved! ({best_acc:.2f}%)")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")

# Save final
torch.save(model.state_dict(),
           os.path.join(MODELS_PATH, "final_cnn_model.pth"))
joblib.dump({'num_classes': NUM_CLASSES, 'img_size': 48, 'best_acc': best_acc},
            os.path.join(MODELS_PATH, "cnn_config.pkl"))

# ── PLOTS ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ep = range(1, EPOCHS+1)
axes[0].plot(ep, train_losses, 'b-o', markersize=4, label='Train')
axes[0].plot(ep, test_losses,  'r-o', markersize=4, label='Test')
axes[0].set_title('Fine-tuning Loss',      fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, train_accs, 'b-o', markersize=4, label='Train')
axes[1].plot(ep, test_accs,  'r-o', markersize=4, label='Test')
axes[1].set_title('Fine-tuning Accuracy',  fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 100)

fig.suptitle(f'MobileNetV2 Fine-tuning (Best: {best_acc:.2f}%)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "13b_cnn_finetuning_curves.png"), dpi=150)
plt.close()
print("Saved: 13b_cnn_finetuning_curves.png")

# Confusion matrix
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(DEVICE)
        _, pred = torch.max(model(imgs), 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(lbls.numpy())

cm     = confusion_matrix(all_labels, all_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=['Baseline','Stress','Amusement'],
            yticklabels=['Baseline','Stress','Amusement'], ax=ax)
ax.set_title(f'CNN Confusion Matrix — Fine-tuned ({best_acc:.2f}%)',
             fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "14b_cnn_finetuned_confusion.png"), dpi=150)
plt.close()
print("Saved: 14b_cnn_finetuned_confusion.png")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
      target_names=['Baseline','Stress','Amusement']))
print("\n✅ Fine-tuning complete!")