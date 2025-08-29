import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
sys.path.append('/home/diane.kim/ACS/tabular/model/TabPFN/src')
sys.path.append('/mnt/rcl-server/workspace/diane/nature')

from model.data_preprocessor import *
from model.fix_seed import *
from model.tabpfn import tabpfn_search, eval

if __name__ == "__main__":
    # print("Job initiated")
    # if len(sys.argv) > 1:
    #     seed = int(sys.argv[1])
    # else:
    #     seed = 0  # fallback default if not passed

    seed=0

    print(f"Running with seed: {seed}")
    fix_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor = load_data(binary=True)

    best_params, best_model = tabpfn_search(X_train, y_train, seed=seed)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba (X_test)

    out = eval(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        set_name="test",
        verbose=True
    )