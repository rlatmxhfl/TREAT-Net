import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.append('/home/diane.kim/ACS/tabular/model/TabPFN/src')
sys.path.append('/mnt/rcl-server/workspace/diane/nature')
sys.path.append('/mnt/rcl-server/workspace/diane/nature/baseline/tabpfn')

from tabpfn.classifier import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.config import ModelInterfaceConfig

from model.data_preprocessor import *
from model.fix_seed import *
from param_tuning.tabpfn_tuning import tabpfn_search, eval

if __name__ == "__main__":
    # print("Job initiated")
    # if len(sys.argv) > 1:
    #     seed = int(sys.argv[1])
    # else:
    #     seed = 0  # fallback default if not passed

    seed=21

    print(f"Running with seed: {seed}")
    fix_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor = load_data(binary=True)

    # best_params, best_model = tabpfn_search(X_train, y_train, seed=seed)

    best_model = TabPFNClassifier(
        n_estimators=1,
        random_state=seed,
        balance_probabilities=True,
        inference_config=ModelInterfaceConfig(
        #   OUTLIER_REMOVAL_STD=1000,
          REGRESSION_Y_PREPROCESS_TRANSFORMS=(None,),
          FINGERPRINT_FEATURE=False,
          PREPROCESS_TRANSFORMS=(PreprocessorConfig("none",),))
    )

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba (X_test)

    out = eval(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        set_name="test",
        verbose=True
    )