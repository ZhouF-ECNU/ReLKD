#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=3 python train_with_coarse.py \
    --dataset_name 'aircraft' \
    --setting 'default' \
    --batch_size 256 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --warmup_coarse_weight_start_epoch 30 \
    --warmup_coarse_weight_end_epoch 60 \
    --coarse_weight 0.8 \
    --memax_weight 1 \
    --use_coarse_label 'False' \
    --use_memory_queue 'True' \
    --mq_start_add_epoch 0 \
    --mq_start_query_epoch 10 \
    --mq_query_mode 'soft' \
    --mq_maxsize 1024 \
    --use_prototypes_attention 'False' \
    --use_gt_coarse_label 'False' \
    --exp_name 'aircraft_default_ReLKD'
