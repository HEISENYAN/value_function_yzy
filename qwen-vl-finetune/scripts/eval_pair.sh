#!/bin/bash

llm=/mnt/lijunhao/value_function/checkpoints/Qwen2.5-VL-3B-Instruct
datasets="/mnt/lijunhao/robotwin/adjust_bottle/aloha-agilex_clean_50"
output_dir=/mnt/lijunhao/value_function/eval_output/v1
max_episodes=50

python qwenvl/train/eval_qwen_pair.py \
    --model_name_or_path ${llm} \
    --eval_dataset_use ${eval_datasets} \
    --pair_mode True \
    --pair_short_step 8 \
    --pair_mid_step 16 \
    --pair_random_min 1 \
    --output_dir ${output_dir} \
    --max_episodes ${max_episodes}
