#!/bin/bash

# List of seeds
seeds=(0 12 21 7 15)
# seeds=(7)

for seed in "${seeds[@]}"
do
  echo "Launching seed $seed ..."
  nohup python tabpfn_mlp_main.py $seed > tabpfn_mlp_seed${seed}.log 2>&1 &
done

echo "All runs launched in background."
# echo "Use 'tail -f tabonlyseedX.log' to monitor logs."