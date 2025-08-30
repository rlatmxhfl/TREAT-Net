import os

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

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

def class_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    out = {}
    for c in classes:
        mask = (y_true == c)
        denom = mask.sum()
        out[int(c) if hasattr(c, "__init__") else c] = (float((y_pred[mask] == c).sum()/denom) if denom else np.nan)
    return out

def _sens_at_spec(y_true, y_scores, targets=(0.2, 0.4, 0.6)):
    """Return dict: spec_target -> (sensitivity, specificity, threshold). Ex. {0.2: (sens, spec, thr), 0.4: (...), 0.6: (...)}"""
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    spec = 1.0 - fpr
    out = {}
    for t in targets:
        idxs = np.where(spec>=t)[0] #unwraps the tuple array
        if len(idxs) == 0:
            out[t] = (np.nan, np.nan, np.nan)
        else:
            i = idxs[-1]
            out[t] = (tpr[i], spec[i], thr[i])
    return out

def model_eval(y_true, y_pred, y_proba, positive_class=1, specificities=(0.2, 0.4, 0.6), 
         average="weighted", verbose=True, set_name=None):
    
    y_true = np.asarray(y_true)
    
    scores = None

    y_proba=np.asarray(y_proba)

    if y_proba.ndim == 1: #already positive-class probability
        scores = y_proba #shape (N,)
    elif y_proba.ndim == 2:
        if y_proba.shape[1] == 1:
            scores = y_proba.ravel()  #shape (N,1) -> (N,)
            # .ravel() takes a multi-dimensional array and flattens to 1D (e.g., (2,3) to (6,)) - could be better than .flatten() because it works directly on the original data whenever possible
        elif y_proba.shape[1] == 2:
            scores = y_proba[:, 1] #posititive class column
        else:
            print("### multiclass probs (N,C>2): you can set scores=None (AUC handled separately). scores is set to None ###")
            scores=None
    else:
        scores = y_proba.ravel() #flatten to 1D if possible

    y_pred = np.asarray(y_pred)

    metrics = {
        "ROC AUC": (roc_auc_score(y_true, scores) if scores is not None else np.nan),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Class Accuracy": class_accuracy(y_true, y_pred),
    }

    sens_at_spec = {}
    if scores is not None and len(np.unique(y_true)) == 2:
        sens_at_spec = _sens_at_spec(y_true, scores, specificities)
    
    result = {
        "metrics": metrics,
        "sensitivities": sens_at_spec
    }

    if verbose:
        head = []
        if set_name: head.append(set_name.upper())
        # if epoch is not None: head.append(f"Epoch {epoch}")
        if head: print("[" + " | ".join(head) + "]")
        for k, v in metrics.items():
            if k == "Class Accuracy":
                print("Class Accuracy:")
                for cls, acc in v.items():
                    print (f"  {cls}: {acc:.4f}" if acc == acc else f"  {cls}: nan")
            else: 
                print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")
        if sens_at_spec:
            for t, (sens, spec, thr) in sens_at_spec.items():
                pct = int(t*100)
                print(f"Spec {pct}% → Sens: {sens:.4f}, Spec: {spec:.4f}, Thr: {thr:.4f}")
        print()

    return result