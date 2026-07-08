"""Shared data-loading helpers for the biosignal feature pipeline."""
import os
from collections import namedtuple

import pandas as pd

from src.constants import META_COLS, LABEL_COL, SUBJECT_COL

# X: feature matrix (metadata dropped); y: integer labels; groups: subject IDs
# (for subject-aware CV); feature_names: ordered feature-column names for X.
FeatureData = namedtuple("FeatureData", ["X", "y", "groups", "feature_names"])


def load_features(processed_dir, filename="features.csv"):
    """Load features.csv and split it into model inputs plus metadata.

    Reads ``<processed_dir>/<filename>`` and separates the numeric feature
    columns from the two metadata columns (``label``, ``subject_id``). The
    metadata is never included in the feature matrix, so ``subject_id`` cannot
    leak into a model as a feature. Column order is preserved, so the returned
    ``feature_names`` line up with the columns of ``X``.

    Parameters
    ----------
    processed_dir : str | pathlib.Path
        Directory containing the features CSV (typically ``DATA_DIR``).
    filename : str, optional
        CSV file name, by default ``"features.csv"``.

    Returns
    -------
    FeatureData
        Named tuple ``(X, y, groups, feature_names)`` where ``X`` is the
        feature matrix (ndarray), ``y`` the integer labels (ndarray),
        ``groups`` the per-row subject IDs (ndarray, for LOSO / grouped CV),
        and ``feature_names`` the ordered list of feature-column names.
    """
    df = pd.read_csv(os.path.join(processed_dir, filename))
    feature_names = [c for c in df.columns if c not in META_COLS]
    X = df[feature_names].values
    y = df[LABEL_COL].values
    groups = df[SUBJECT_COL].values
    return FeatureData(X, y, groups, feature_names)
