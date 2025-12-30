#!/bin/bash

# Model configuration
model_name_or_path=./output  # Base model path

# Dataset configuration
# datasets=qwen-vl-finetune/data/openpi/merged_beat_block_hammer.pkl
datasets=qwen-vl-finetune/data/RoboTwin/dataset/beat_block_hammer/aloha-agilex_clean_50

# ValueTokenizer configuration
value_tokenizer_bins=201  # Number of bins (should match model's extra tokens)
value_tokenizer_min=-1.0  # Minimum value
value_tokenizer_max=0.0  # Maximum value

# Evaluation configuration
output_dir=./eval_output
max_episodes=1  # Number of episodes to evaluate

# Launch evaluation
python qwenvl/train/eval_qwen.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_use ${datasets} \
    --value_tokenizer_bins ${value_tokenizer_bins} \
    --value_tokenizer_min ${value_tokenizer_min} \
    --value_tokenizer_max ${value_tokenizer_max} \
    --output_dir ${output_dir} \
    --max_episodes ${max_episodes} \

