#!/bin/bash

#SBATCH -J stage0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time 1:00:00
#SBATCH --qos=m5
#SBATCH --gres=gpu:rtx6000:1
##SBATCH --time 24:00:00
##SBATCH --account=deadline
##SBATCH --qos=deadline
##SBATCH --gres=gpu:a40:1
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=1-5
#SBATCH --exclude=gpu144
# echo 

#export WANDB_RUN_ID=$SLURM_JOB_ID

# seed=$SLURM_ARRAY_TASK_ID
datetime=$(date +"%Y-%m-%d_%H-%M-%S")

seeds=(0 12 21 7 15)

# if [ -z "$seed" ]; then
#   seed=0 #
# fi

if [ -z ${device} ]
then
  device=3
fi
export CUDA_VISIBLE_DEVICES=${device}

for seed in "${seeds[@]}"; do
  echo "Running with seed ${seed}."

  nohup python echoprime.py \
    --exp_name min_len \
    --num_workers 0 \
    --batch_size 32 \
    --eval_batch_size 1 \
    --wdb_group Jun27_sgd_1000eps \
    -lr 1.e-4 \
    -wd 1.e-2 \
    --num_layers 2 \
    --nhead 4 \
    --optim sgd \
    --target tp \
    --epochs 1000 \
    --loss_fn ce \
    --seed ${seed} \
    --exp_dir /home/diane.kim/nature/baseline/echoprime/checkpoint/$datetime \
    --mode video \
    $@ > echprime_seed${seed}.log 2>&1 &
done