#!/bin/bash

DATA=../data
TRAINER=AdaCPI
DATASET=$1
W=$2
LOADEP=$3

CFG=vit_b16_c2_ep20_batch4_2ctx
SHOTS=16
SUB=new
ADAPTIVE_FUSION=True
FUSION_DEPTH=None
NUM_WORKERS=10
TEMPERATURE=0.5 #fixed for evaluation

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/${SHOTS}shot/seed${SEED}
    MODEL_DIR=./output/${TRAINER}/base2new/train_base/${COMMON_DIR}
    DIR=./output/${TRAINER}/base2new/test_${SUB}/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "Evaluating model"
        echo "Results are available in ${DIR}. Resuming..."
        python ./train_adaptive.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file ./configs/datasets/${DATASET}.yaml \
        --config-file ./configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        --no-train \
        DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
        TRAINER.AdaCPI.W ${W} \
        TRAINER.AdaCPI.FUSION_DEPTH ${FUSION_DEPTH} \
        TRAINER.AdaCPI.ADAPTIVE_FUSION ${ADAPTIVE_FUSION} \
        TRAINER.AdaCPI.TEMPERATURE ${TEMPERATURE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    else
        echo "Evaluating model"
        echo "Runing the first phase job and save the output to ${DIR}"
        python ./train_adaptive.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file ./configs/datasets/${DATASET}.yaml \
        --config-file ./configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        --no-train \
        DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
        TRAINER.AdaCPI.W ${W} \
        TRAINER.AdaCPI.FUSION_DEPTH ${FUSION_DEPTH} \
        TRAINER.AdaCPI.ADAPTIVE_FUSION ${ADAPTIVE_FUSION} \
        TRAINER.AdaCPI.TEMPERATURE ${TEMPERATURE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done

 