#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ENTROPIES="0.23 0.28 0.46"

for ENTROPY in $ENTROPIES; do
    echo $ENTROPY
    python ../examples/masked_run_highway_glue.py --model_type albert \
      --model_name_or_path ./saved_models/masked_albert/QNLI/bertarized_two_stage_pruned_0.5 \
      --task_name QNLI \
      --do_eval \
      --do_lower_case \
      --data_dir ./glue_data/QNLI \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size=1 \
      --overwrite_output_dir \
      --output_dir ./saved_models/masked_albert/QNLI/bertarized_two_stage_pruned_0.5  \
      --plot_data_dir ./plotting/ \
      --early_exit_entropy $ENTROPY \
      --eval_highway \
      --overwrite_cache
done
