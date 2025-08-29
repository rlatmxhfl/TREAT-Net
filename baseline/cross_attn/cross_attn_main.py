import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
sys.path.append('/mnt/rcl-server/workspace/diane/nature')
sys.path.append('/mnt/rcl-server/workspace/diane/nature/baseline')

from cross_attn import *

if __name__ == "__main__":
    main(parse_args())