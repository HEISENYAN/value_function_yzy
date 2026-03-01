#!/bin/bash
export WANDB_API_KEY=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
export WANDB_PROJECT="value-function"
export WANDB_ENTITY="heisen0928-the-hong-kong-polytechnic-university"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}


deepspeed=./scripts/zero3.json
entry_file=qwenvl/train/train_qwen_pair.py

llm=Qwen/Qwen2.5-VL-3B-Instruct
# datasets="/mnt/lijunhao/robotwin/adjust_bottle/aloha-agilex_clean_50"
datasets="/mnt/bn/ic-vlm/zhufangqi/code/datasets/robotwin-50-50-clean"

run_name="robotwin_clean_50_pair_v1"
output_dir=./output/${run_name}
log_dir=${output_dir}/runs

lr=1e-5
value_head_lr=5e-5
batch_size=2
grad_accum_steps=4

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --pair_mode True \
    --pair_short_step 8 \
    --pair_mid_step 16 \
    --pair_random_min 1 \
    --pair_add_backward True \
    --pair_use_t_group_weight True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --max_steps 100000 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_strategy steps \
    --eval_steps 100000 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --value_head_lr ${value_head_lr} \
    --value_head_weight_decay 0.0 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --logging_dir ${log_dir} \
    --report_to tensorboard \
    --logging_first_step True"

torchrun --nproc_per_node=8 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
