#!/bin/bash

# Model configuration
export run_name="robotwin_clean_50v1"
export step_num=10000

llm=/mnt/bn/ic-vlm/zhufangqi/code/value_function/qwen-vl-finetune/output/${run_name}/checkpoint-${step_num}  # Base model path
cp /mnt/bn/ic-vlm/zhufangqi/code/value_function/qwen-vl-finetune/tools/checkpoints/Qwen2.5-VL-3B-Instruct-resize/preprocessor_config.json ${llm}
cp /mnt/bn/ic-vlm/zhufangqi/code/value_function/qwen-vl-finetune/tools/checkpoints/Qwen2.5-VL-3B-Instruct-resize/video_preprocessor_config.json ${llm}
# Dataset configuration
eval_datasets="/mnt/bn/ic-vlm/zhufangqi/code/datasets/robotwin-50-50-clean"
#eval_datasets="/mnt/bn/ic-vlm/zhufangqi/code/.cache/huggingface/lerobot/beat_block_hammer_rollout"
# eval_datasets=/project/peilab/lerobot/folding_clothes

# ValueTokenizer configuration
value_tokenizer_bins=201  # Number of bins
value_tokenizer_min=-1.0  # Minimum value
value_tokenizer_max=0.0  # Maximum value

# Evaluation configuration
output_dir=./eval_output/${run_name}
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

