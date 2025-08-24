#!/bin/bash

set -x

MODEL_PATH=mistralai/Mistral-7B-v0.3

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --stage pt \
    --do_train \
    --dataset augmentoolkit_pretrain \
    --cutoff_len 8192 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/mistral-7b-v0.3/full/pt \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000