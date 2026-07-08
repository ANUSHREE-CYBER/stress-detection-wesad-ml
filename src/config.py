"""Central path configuration for the Stress Detection project.

All project-internal paths (processed data, trained models, generated plots)
are resolved relative to the repository root, so the code runs unchanged on
any machine after cloning — no absolute paths are baked in.

The two RAW datasets (WESAD and FER2013) are large and git-ignored, but on
this machine they live inside the repo under ``WESAD Dataset/``. Anyone
running this project should either:

  1. Set the ``WESAD_DATA_PATH`` and ``FER2013_DATA_PATH`` environment
     variables to point at their local copies of the datasets, or
  2. Place the datasets under ``<repo root>/WESAD Dataset/`` so the defaults
     below resolve correctly.

The defaults are resolved relative to the repo root and are only used as a
fallback when the environment variables are not set.
"""
import os
from pathlib import Path

# ── PROJECT-INTERNAL PATHS (relative to the repo root) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR  = DATA_DIR / "plots"

# ── EXTERNAL DATASET PATHS (override via environment variables) ────────────
WESAD_DIR   = Path(os.environ.get("WESAD_DATA_PATH",   PROJECT_ROOT / "WESAD Dataset" / "WESAD"))
FER2013_DIR = Path(os.environ.get("FER2013_DATA_PATH", PROJECT_ROOT / "WESAD Dataset" / "FER2013"))
