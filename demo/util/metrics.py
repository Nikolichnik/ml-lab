"""
Metrics for classification tasks.
"""

import numpy as np

from sklearn.metrics import roc_auc_score, f1_score

import matplotlib.pyplot as plot


def multiclass_roc_auc(
    probs: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Compute the multiclass ROC AUC score.
    
    Args:
        probs (numpy.ndarray): Predicted probabilities for each class, shape (N, C).
        targets (numpy.ndarray): True class labels, shape (N,).

    Returns:
        float: ROC AUC score.
    """
    # probs: (N, C) numpy; targets: (N,) numpy
    y_true = targets
    y_prob = probs

    try:
        return roc_auc_score(y_true, y_prob, multi_class="ovr")
    except ValueError:
        return np.nan

def macro_f1(predictions, targets) -> float:
    """
    Compute the macro F1 score.

    Args:
        predictions (numpy.ndarray): Predicted class labels, shape (N,).
        targets (numpy.ndarray): True class labels, shape (N,).

    Returns:
        float: Macro F1 score.
    """
    return f1_score(targets, predictions, average="macro")

def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, np.ndarray, list, list]:
    """
    Compute the Expected Calibration Error (ECE).

    Args:
        probabilities (numpy.ndarray): Predicted probabilities for each class, shape (N, C).
        targets (numpy.ndarray): True class labels, shape (N,).
        n_bins (int): Number of bins to use.

    Returns:
        tuple: ECE value, bin edges, bin accuracies, bin confidences.
    """
    # Reliability diagram + ECE
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    bin_acc, bin_conf = [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])

        if mask.any():
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
            bin_acc.append(acc)
            bin_conf.append(conf)
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)

    return ece, bins, bin_acc, bin_conf

def plot_reliability_diagram(
    bins: np.ndarray,
    bin_acc: list,
    bin_conf: list,
    title="Reliability Diagram",
) -> None:
    """
    Plot the reliability diagram.
    
    Args:
        bins (numpy.ndarray): Bin edges.
        bin_acc (list): Accuracy per bin.
        bin_conf (list): Confidence per bin.
        title (str): Title of the plot.

    Returns:
        None
    """
    plot.figure(figsize=(5,5))
    xs = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    plot.plot([0,1],[0,1], linestyle="--")
    plot.scatter(xs, bin_acc, label="Accuracy per bin")
    plot.plot(xs, bin_conf, label="Confidence per bin")
    plot.xlabel("Confidence")
    plot.ylabel("Accuracy/Confidence")
    plot.title(title)
    plot.legend()
    plot.tight_layout()

    plot.show()
