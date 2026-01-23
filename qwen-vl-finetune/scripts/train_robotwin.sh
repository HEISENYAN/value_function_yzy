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
# datasets=/home/teamcommon/.cache/huggingface/lerobot/beat_block_hammer
# datasets="/path/to/dataset1,/path/to/dataset2,/path/to/dataset3"

datasets="/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/adjust_bottle_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/beat_block_hammer_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/blocks_ranking_rgb_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/blocks_ranking_size_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/click_alarmclock_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/click_bell_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/dump_bin_bigbin_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/grab_roller_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/handover_block_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/handover_mic_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/hanging_mug_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/lift_pot_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/move_can_pot_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/move_pillbottle_pad_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/move_playingcard_away_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/move_stapler_pad_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_can_basket_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_cans_plasticbox_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_container_plate_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_dual_shoes_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_empty_cup_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_fan_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_mouse_pad_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_object_basket_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_object_scale_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_object_stand_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_phone_stand_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/place_shoe_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/press_stapler_randomized_500,/project/peilab/junhao/.cache/huggingface/lerobot/robotwin_aloha_lerobot/put_bottles_dustbin_randomized_500"
eval_datasets=/project/peilab/lerobot/beat_block_hammer_rollout

# ValueTokenizer configuration (for continuous values like action values)
use_value_tokenizer=True  # Enable ValueTokenizer
value_tokenizer_bins=201  # Number of bins
value_tokenizer_min=-1.0  # Minimum value
value_tokenizer_max=0.0  # Maximum value

# Output configuration
export WANDB_API_KEY=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
export WANDB_PROJECT="value_function"
export WANDB_ENTITY="fcdljh"

run_name="robotwin_hard_30"
output_dir=./output/robotwin_hard_30

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
    --max_steps 10000 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --image_size 224 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --use_value_tokenizer ${use_value_tokenizer} \
    --value_tokenizer_bins ${value_tokenizer_bins} \
    --value_tokenizer_min ${value_tokenizer_min} \
    --value_tokenizer_max ${value_tokenizer_max} \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=8 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

# python -m debugpy --listen localhost:5678 --wait-for-client ${entry_file} ${args}