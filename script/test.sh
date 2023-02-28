#!/bin/bash

# Specs.
SEED=2021
FOLD=0  # testing on this fold
DATA=data/BraTS
PRETRAINED=results/train/fold${FOLD}/model.pth
SAVE_FOLDER=results/test/fold${FOLD}
mkdir -p ${SAVE_FOLDER}

# Run.
python main_test.py \
--data_dir ${DATA} \
--save_dir ${SAVE_FOLDER} \
--pretrained_root "${PRETRAINED}" \
--dataset BraTS \
--fold ${FOLD} \
--seed ${SEED}

# Note: EP2 is default, for EP1 set --EP1 True, --n_shot 3.

