#!/bin/bash

# custom config
DATA=./data
TRAINER=AdaCPI

DATASET=$1
W=$2 #lambda
EPOCH=$3

CFG=vit_b16_c2_ep20_batch4_2ctx
ADAPTIVE_FUSION=True
FUSION_DEPTH=None
NUM_WORKERS=10
SHOTS=16

for SEED in 1 2 3
do
    DIR=./output/${TRAINER}/base2new/train_base/${DATASET}/${SHOTS}shot/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming..."
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
        DATASET.SUBSAMPLE_CLASSES base \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE}\
        DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
        OPTIM.MAX_EPOCH ${EPOCH} \
        OPTIM1.MAX_EPOCH ${EPOCH}
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=0 python ./train_adaptive.py \
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
        DATASET.SUBSAMPLE_CLASSES base \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE}\
        DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
        OPTIM.MAX_EPOCH ${EPOCH} \
        OPTIM1.MAX_EPOCH ${EPOCH}
    fi
done