#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ../examples/run_squad.py \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --task_name SQUADV2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./glue_data/SQUADV2 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 1 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --save_steps 0 \
  --seed 42 \
  --output_dir ./saved_models/albert-base/SQUADV2/teacher \
  --overwrite_cache \
  --overwrite_output_dir
