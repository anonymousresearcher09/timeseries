#!/bin/bash
# Training script for AmbiguousMIL with Contrastive Learning
# Usage: bash run_train.sh
# Multi-GPU (DDP): CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 main_cl_exp.py [args]

# UCR/UEA benchmark datasets
DATASETS=(
  "ArticularyWordRecognition"
  "BasicMotions"
  "Cricket"
  "DuckDuckGeese"
  "Epilepsy"
  "ERing"
  "FaceDetection"
  "FingerMovements"
  "HandMovementDirection"
  "Handwriting"
  # "Heartbeat"
  "Libras"
  "LSST"
  "MotorImagery"
  "NATOPS"
  "PenDigits"
  "PEMS-SF"
  "PhonemeSpectra"
  "RacketSports"
  "SelfRegulationSCP1"
  "SelfRegulationSCP2"
  "StandWalkJump"
  "UWaveGestureLibrary"
  "JapaneseVowels"
  # "dba"
)

GPU_IDS="0,1"
N_GPU=2
PORT=29500
MODEL="AmbiguousMIL"
DATATYPE="mixed"
NUM_EPOCHS=1500
EPOCH_DES=20

for DATASET in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Running ${MODEL} on dataset: ${DATASET}"
  echo "========================================"
  CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
    --nproc_per_node=${N_GPU} \
    --master_port=${PORT} \
    main_cl_exp.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --datatype ${DATATYPE} \
    --num_epochs ${NUM_EPOCHS} \
    --epoch_des ${EPOCH_DES} \
    --bag_loss_w 0.5 \
    --inst_loss_w 0.2 \
    --sparsity_loss_w 0.05 \
    --proto_loss_w 0.2 \
    --proto_tau 0.1 \
    --proto_sim_thresh 0.5 \
    --proto_win 5
done
