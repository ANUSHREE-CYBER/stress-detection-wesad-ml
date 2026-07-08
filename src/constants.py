"""Shared constants for the stress-detection pipeline.

Central definitions for the 3-class label scheme and its display colours, so
the same mappings are not copy-pasted across the notebooks. Import from here
rather than redefining ``LABEL_NAMES`` / colour dicts locally.
"""

# Canonical class ordering: 0 = Baseline, 1 = Stress, 2 = Amusement
LABEL_NAMES = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}
# List form (label-index order) for APIs that want positional class names,
# e.g. sklearn's ``target_names`` or seaborn tick labels.
LABEL_NAMES_LIST = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]

# Per-class display colours (matplotlib hex), keyed the same way.
CLASS_COLORS = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
CLASS_COLORS_LIST = [CLASS_COLORS[i] for i in range(len(CLASS_COLORS))]

# Non-feature metadata columns carried through features.csv. These are NEVER
# fed to a model — dropping them keeps subject_id from leaking in as a feature.
LABEL_COL = 'label'
SUBJECT_COL = 'subject_id'
META_COLS = [LABEL_COL, SUBJECT_COL]

# Global RNG seed used across models / cross-validation for reproducibility.
RANDOM_STATE = 42
