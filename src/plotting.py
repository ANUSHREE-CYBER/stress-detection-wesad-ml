"""Shared plotting helpers for the stress-detection pipeline."""
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(ax, y_true, y_pred, class_names, cbar=True, cmap='Blues'):
    """Draw a row-normalised (% of actual) confusion matrix onto an axis.

    Computes the confusion matrix from ``y_true`` / ``y_pred``, converts each
    row to a percentage of that actual class, and renders it as an annotated
    seaborn heatmap on the supplied axis. The x/y axis labels are set to
    'Predicted' / 'Actual'; the caller is responsible for the title.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis to draw on.
    y_true, y_pred : array-like
        True and predicted integer labels.
    class_names : list of str
        Tick labels for both axes, in label-index order.
    cbar : bool, optional
        Whether to draw the heatmap colour bar, by default True.
    cmap : str, optional
        Matplotlib / seaborn colormap name, by default 'Blues'.

    Returns
    -------
    numpy.ndarray
        The raw (count) confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=cbar)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return cm
