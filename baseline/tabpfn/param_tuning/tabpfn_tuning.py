import sys
sys.path.append('/home/diane.kim/ACS/tabular/model/TabPFN/src')

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np

from tabpfn.classifier import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.config import ModelInterfaceConfig

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

def tabpfn_search(X, y, seed=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    softmax_temperature = list(np.linspace(0.1, 1, num=10))
    n_estimators = list(np.linspace(2, 100, num=50, dtype=int))
    balance_probabilities = [True, False]
    average_before_softmax = [True, False]

    random_grid = {'softmax_temperature': softmax_temperature,
                   'n_estimators': n_estimators,
                   'balance_probabilities': balance_probabilities,
                   'average_before_softmax': average_before_softmax,
    }

    PFN = TabPFNClassifier(
        random_state=seed,
        # categorical_features_indices=categorical_column_indexes,
        balance_probabilities=True,
        inference_config=ModelInterfaceConfig(
          OUTLIER_REMOVAL_STD=1000,
          REGRESSION_Y_PREPROCESS_TRANSFORMS=(None,),
          FINGERPRINT_FEATURE=False,
          PREPROCESS_TRANSFORMS=(PreprocessorConfig("none",),)
        )
    )

    search = RandomizedSearchCV(
        estimator=PFN, 
        param_distributions = random_grid, 
        n_iter=50, 
        cv=skf,
        scoring='balanced_accuracy',
        random_state=seed,
        error_score='raise',
        verbose=1,
        refit=True
    )

    # search.fit(X, y)

    print("Best balanced accuracy (CV mean):", round(search.best_score_, 2))
    print("Best params:", search.best_params_)

    best_model = search.best_estimator_

    return search.best_params_, best_model

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

def eval(y_true, y_pred, y_proba, positive_class=1, specificities=(0.2, 0.4, 0.6), 
         average="weighted", verbose=True, set_name=None, epoch=None):
    
    y_true = np.asarray(y_true)
    
    scores = None

    y_proba=np.asarray(y_proba)
    if y_proba.ndim == 2:
        scores = y_proba[:, 1]
    else:
        scores = y_proba.ravel()
    
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
        if epoch is not None: head.append(f"Epoch {epoch}")
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
                print(f"Spec @ {pct}% → Sens: {sens:.4f}, Spec: {spec:.4f}, Thr: {thr:.4f}")
        print()

    return result

    # how to use:
    # # y_true: shape (n,)
    # # y_pred: shape (n,) predicted labels
    # # y_proba: shape (n, 2) or (n,) scores/probs
    # out = eval_classifier(y_test, y_pred, y_proba, set_name="test", epoch=3)

    # # Access pieces:
    # out["metrics"]["Balanced Accuracy"]
    # out["metrics"]["Class Accuracy"]
    # out["sensitivity_at_specificities"][0.6]   # (sens, spec, thr) at 60%