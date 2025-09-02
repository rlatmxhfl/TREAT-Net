import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.append('/home/diane.kim/nature/model')
sys.path.append('/home/diane.kim/nature/model/baseline/tabpfn_mlp')

from utils.data_preprocessor import *
from utils.fix_seed import *
from tabpfn_mlp_bce import train_model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0  # fallback default if not passed

    print(f"Running with seed: {seed}")
    fix_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor = load_data(binary=True, seed=seed, ref_df="/home/diane.kim/nature/data/final/MASTER_2262_wTTE.csv")

    model = train_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes=2, 
                num_epochs=100, optimizer_="SGD", seed=seed, device="cuda")


