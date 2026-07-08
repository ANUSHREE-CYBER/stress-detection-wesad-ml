import os
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import WESAD_DIR

dataset_path = WESAD_DIR

print("=== FILES FOUND ===")
for root, dirs, files in os.walk(dataset_path):
    for f in files:
        full = os.path.join(root, f)
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"{full}  [{size_mb:.2f} MB]")

print("\n=== TRYING TO LOAD FIRST .pkl ===")
for root, dirs, files in os.walk(dataset_path):
    for f in files:
        if f.endswith(".pkl"):
            path = os.path.join(root, f)
            print(f"Loading: {path}")
            with open(path, "rb") as file:
                data = pickle.load(file, encoding="latin1")
            print(f"Type: {type(data)}")
            if isinstance(data, dict):
                print(f"Top-level keys: {list(data.keys())}")
                for k, v in data.items():
                    print(f"  Key: {k}, Type: {type(v)}")
                    if isinstance(v, dict):
                        print(f"    Sub-keys: {list(v.keys())}")
            break
    else:
        continue
    break