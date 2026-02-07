#!/bin/bash
export NCCL_DEBUG=INFO                  # 必须！看到真正 NCCL 报什么（timeout? internal error? connection closed?）
export NCCL_IB_TIMEOUT=100              # 或直接 200/300，解决大部分“等太久”的情况
export NCCL_ASYNC_ERROR_HANDLING=1      # 让 NCCL 更早抛异常而不是 silent hang
export TORCH_NCCL_BLOCKING_WAIT=1       # 配合上面，强制阻塞等待，便于 debug
export NCCL_P2P_DISABLE=1               # 先关 P2P 试试（常见绕过硬件问题
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/mnt/bn/ic-vlm/zhufangqi/code/value_function/qwen-vl-finetune/tools/checkpoints/Qwen2.5-VL-3B-Instruct-resize  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=2
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
# datasets=/home/teamcommon/.cache/huggingface/lerobot/beat_block_hammer
# datasets="/path/to/dataset1,/path/to/dataset2,/path/to/dataset3"
export XDG_CACHE_HOME="/mnt/bn/ic-vlm/zhufangqi/code/.cache/"
export HF_HOME=/mnt/bn/ic-vlm/zhufangqi/code/.cache/huggingface
export HF_HUB_CACHE=/mnt/bn/ic-vlm/zhufangqi/code/.cache/huggingface

datasets="/mnt/bn/ic-vlm/zhufangqi/code/datasets/robotwin-50-50-clean"
eval_datasets=/mnt/bn/ic-vlm/zhufangqi/code/datasets/robotwin-50-50-clean

# ValueTokenizer configuration (for continuous values like action values)
use_value_tokenizer=True  # Enable ValueTokenizer
value_tokenizer_bins=201  # Number of bins
value_tokenizer_min=-1.0  # Minimum value
value_tokenizer_max=0.0  # Maximum value

# Output configuration

export WANDB_API_KEY=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
export WANDB_PROJECT="value-function"
export WANDB_ENTITY="heisen0928-the-hong-kong-polytechnic-university"

run_name="robotwin_clean_50v1"
output_dir=./output/${run_name}

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
    --max_steps 100000 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --image_size 224 \
    --eval_strategy "no" \
    --dataloader_drop_last True \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 20 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --use_value_tokenizer ${use_value_tokenizer} \
    --value_tokenizer_bins ${value_tokenizer_bins} \
    --value_tokenizer_min ${value_tokenizer_min} \
    --value_tokenizer_max ${value_tokenizer_max} \
    --run_name ${run_name} \
    --report_to wandb"

#Launch training
torchrun --nproc_per_node=8 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
     ${entry_file} ${args}

#python -m debugpy --listen localhost:5678 --wait-for-client ${entry_file} ${args}
#bash scripts/train_robotwin_rollout.sh