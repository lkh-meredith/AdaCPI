#!/bin/bash

# custom config
DATA=./data
TRAINER=AdaCPI

DATASET=$1
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)

CFG=vit_b16_c2_ep20_batch4_2ctx  # config file
SUBSAMPLES=all # all, base, new, ten
NUM_WORKERS=10
ADAPTIVE_FUSION=True #the number of CPI layer
FUSION_DEPTH=None
W=0.1 #lambda
for SEED in 1 2 3
do
    DIR=./output/${TRAINER}/${DATASET}/seed${SEED}/
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=0 python ./train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ./configs/datasets/${DATASET}.yaml \
    --config-file ./configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.AdaCPI.W ${W} \
    TRAINER.AdaCPI.FUSION_DEPTH ${FUSION_DEPTH} \
    TRAINER.AdaCPI.ADAPTIVE_FUSION ${ADAPTIVE_FUSION} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLES} \
    DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
done
