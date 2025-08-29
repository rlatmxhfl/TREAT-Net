import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
sys.path.append('/mnt/rcl-server/workspace/diane/nature')
sys.path.append('/mnt/rcl-server/workspace/diane/nature/baseline')

from model.data_preprocessor import *
from model.fix_seed import *
from cross_attn import *

if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0  # fallback default if not passed

    print(f"Running with seed: {seed}")
    fix_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor = load_data(binary=True, seed=seed)
    
    main(parse_args())