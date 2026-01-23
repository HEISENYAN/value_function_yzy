#!/bin/bash

# Model configuration
llm=./output/robotwin_single_view/checkpoint-3000  # Base model path

# Dataset configuration
eval_datasets=/project/peilab/junhao/.cache/huggingface/lerobot/beat_block_hammer_rollout
# eval_datasets=/project/peilab/lerobot/folding_clothes

# ValueTokenizer configuration
value_tokenizer_bins=201  # Number of bins
value_tokenizer_min=-1.0  # Minimum value
value_tokenizer_max=0.0  # Maximum value

# Evaluation configuration
output_dir=./eval_output
max_episodes=1000  # Number of episodes to evaluate

# Launch evaluation
python qwenvl/train/eval_qwen.py \
    --model_name_or_path ${llm} \
    --eval_dataset_use ${eval_datasets} \
    --value_tokenizer_bins ${value_tokenizer_bins} \
    --value_tokenizer_min ${value_tokenizer_min} \
    --value_tokenizer_max ${value_tokenizer_max} \
    --output_dir ${output_dir} \
    --max_episodes ${max_episodes} \

