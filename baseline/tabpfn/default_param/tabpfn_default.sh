#!/bin/bash

# List of seeds
seeds=(0 12 21 2 15)
# seeds=(7)

for seed in "${seeds[@]}"
do
  echo "Launching seed $seed ..."
  nohup python tabpfn_default_main.py $seed > tabpfn_default_seed${seed}.log 2>&1 &
done

echo "All runs launched in background."
# echo "Use 'tail -f tabonlyseedX.log' to monitor logs."