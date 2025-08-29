import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
sys.path.append('/mnt/rcl-server/workspace/diane/nature')
sys.path.append('/mnt/rcl-server/workspace/diane/nature/baseline/tabpfn_mlp')

from model.data_preprocessor import *
from model.fix_seed import *
from tabpfn_mlp import train_model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0  # fallback default if not passed

    print(f"Running with seed: {seed}")
    fix_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor = load_data(binary=True, seed=seed)

    model = train_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes=2, 
                num_epochs=200, optimizer_="SGD", seed=seed, device="cuda")


