import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import WESAD_DIR

path = WESAD_DIR / "S2" / "S2.pkl"

with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("=== TOP LEVEL KEYS ===")
print(list(data.keys()))

print("\n=== SIGNAL KEYS ===")
for device, signals in data['signal'].items():
    print(f"\nDevice: {device}")
    if isinstance(signals, dict):
        for sig_name, sig_val in signals.items():
            print(f"  {sig_name}: shape={np.array(sig_val).shape}")
    else:
        print(f"  shape={np.array(signals).shape}")

print("\n=== LABEL INFO ===")
labels = data['label']
print(f"Label shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")
print(f"Label counts: { {int(l): int(np.sum(labels==l)) for l in np.unique(labels)} }")

print("\n=== SUBJECT ===")
print(f"Subject ID: {data['subject']}")