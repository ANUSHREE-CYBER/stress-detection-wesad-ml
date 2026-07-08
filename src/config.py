"""Central path configuration for the Stress Detection project.

All project-internal paths (processed data, trained models, generated plots)
are resolved relative to the repository root, so the code runs unchanged on
any machine after cloning — no absolute paths are baked in.

The two RAW datasets (WESAD and FER2013) live OUTSIDE the repository because
they are too large to commit. Anyone running this project should either:

  1. Set the ``WESAD_DATA_PATH`` and ``FER2013_DATA_PATH`` environment
     variables to point at their local copies of the datasets, or
  2. Place the datasets at the default location below.

The defaults point at the original author's machine layout and are only used
as a fallback when the environment variables are not set.
"""
import os
from pathlib import Path

# ── PROJECT-INTERNAL PATHS (relative to the repo root) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR  = DATA_DIR / "plots"

# ── EXTERNAL DATASET PATHS (override via environment variables) ────────────
WESAD_DIR   = Path(os.environ.get("WESAD_DATA_PATH", r"D:\WESAD Dataset\WESAD"))
FER2013_DIR = Path(os.environ.get("FER2013_DATA_PATH", r"D:\WESAD Dataset\FER2013"))
