import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys

from model.src.models import main, parse_args

if __name__ == "__main__":
    main(parse_args())