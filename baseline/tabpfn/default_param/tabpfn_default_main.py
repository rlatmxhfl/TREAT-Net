import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.append('/home/diane.kim/ACS/tabular/model/TabPFN/src')
sys.path.append('/home/diane.kim/nature/model')

from tabpfn.classifier import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.config import ModelInterfaceConfig

from utils.data_preprocessor import *
from utils.fix_seed import *
from src.evaluation import model_eval

if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 21 # fallback default if not passed

    print(f"Running with seed: {seed}")
    fix_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor = load_data(binary=True, seed=seed, ref_df="/home/diane.kim/nature/data/final/MASTER_2262_wTTE.csv")

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

    out = model_eval(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        set_name="test",
        verbose=True
    )