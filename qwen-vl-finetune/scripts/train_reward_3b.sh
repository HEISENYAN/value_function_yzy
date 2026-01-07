#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=./checkpoints/Qwen2.5-VL-3B-Instruct-resize  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=16
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
# 支持多个数据集，用逗号分隔
# datasets=/home/teamcommon/.cache/huggingface/lerobot/beat_block_hammer
# 或者使用多个数据集进行联合训练：
# datasets="/path/to/dataset1,/path/to/dataset2,/path/to/dataset3"
datasets=/home/teamcommon/.cache/huggingface/lerobot/beat_block_hammer_clean_50_video
eval_datasets=/home/teamcommon/.cache/huggingface/lerobot/beat_block_hammer_clean_50_video

# ValueTokenizer configuration (for continuous values like action values)
use_value_tokenizer=True  # Enable ValueTokenizer
value_tokenizer_bins=201  # Number of bins (should match model's extra tokens)
value_tokenizer_min=-1.0  # Minimum value
value_tokenizer_max=0.0  # Maximum value

# Output configuration
run_name="qwen25vl-rm"
output_dir=./output/gpus_8

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --eval_dataset_use ${eval_datasets} \
    --val_ratio 0.1 \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --max_steps 5000 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --image_size 224 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --use_value_tokenizer ${use_value_tokenizer} \
    --value_tokenizer_bins ${value_tokenizer_bins} \
    --value_tokenizer_min ${value_tokenizer_min} \
    --value_tokenizer_max ${value_tokenizer_max} \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training
torchrun --nproc_per_node=8 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

# python -m debugpy --listen localhost:5678 --wait-for-client ${entry_file} ${args}