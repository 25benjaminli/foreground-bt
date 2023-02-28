#!/bin/bash

# Specs.
SEED=2021
FOLD=0  # indicate testing fold (will be trained on the rest!)
RUNS=1  # number of runs (repetitions)
DATA=data/BraTS
SAVE_FOLDER=results/train/fold${FOLD}

# Run.
mkdir -p ${SAVE_FOLDER}
for _ in $(seq 1 ${RUNS})
do
  python main_train.py \
  --data_dir ${DATA} \
  --save_dir ${SAVE_FOLDER} \
  --dataset BraTS \
  --n_sv 5000 \
  --fold ${FOLD} \
  --seed ${SEED}
done

