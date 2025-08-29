import wandb
import numpy as np
import pylab as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

from .misc import DummyRun


def compute_worst_class_accuracy(y_true, y_pred):
    """
    Computes the worst per-class accuracy (i.e., minimum accuracy across all classes).

    Args:
        y_true (np.ndarray): Ground-truth labels (shape: [N])
        y_pred (np.ndarray): Predicted labels (shape: [N])

    Returns:
        float: Worst-class accuracy in percentage (%)
    """
    unique_classes = np.unique(y_true)
    worst_acc = 1

    for cls in unique_classes:
        cls_mask = y_true == cls
        if cls_mask.sum() == 0:
            continue  # skip if class is missing (should not happen in eval)
        cls_acc = (y_pred[cls_mask] == y_true[cls_mask]).mean()
        worst_acc = min(worst_acc, cls_acc)

    return worst_acc


def show_confusion_matrix(y_true, y_pred, set_name, run=DummyRun(), auto_close=True):
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap="Blues",
                # xticklabels=["MANAG", "PTCA", "SURG"],
                # yticklabels=["MANAG", "PTCA", "SURG"]
                xticklabels=["MANAG", "INTERVENTION"],
                yticklabels=["MANAG", "INTERVENTION"]
                )
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    if not isinstance(run, DummyRun):
        run.log({f'confusion_matrix/{set_name}': wandb.Image(fig)}, commit=False)

    if auto_close:
        plt.close()


def safe_auc_score(y_true, y_prob, average='macro', multi_class='ovr'):
    try:
        return 100.0 * roc_auc_score(y_true, y_prob, average=average, multi_class=multi_class)
    except ValueError:
        return float('nan')  # e.g. if only one class present