#!/bin/bash

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

deepspeed=./scripts/zero3.json
entry_file=qwenvl/train/train_qwen_pair.py

llm=/mnt/lijunhao/value_function/checkpoints/Qwen2.5-VL-3B-Instruct
# datasets="/mnt/lijunhao/robotwin/adjust_bottle/aloha-agilex_clean_50"
datasets="/mnt/lijunhao/robotwin/move_can_pot/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_can_basket/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/put_bottles_dustbin/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/adjust_bottle/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/move_pillbottle_pad/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_cans_plasticbox/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/put_object_cabinet/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/beat_block_hammer/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/move_playingcard_away/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_container_plate/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/rotate_qrcode/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/blocks_ranking_rgb/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/move_stapler_pad/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_dual_shoes/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/scan_object/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/blocks_ranking_size/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/open_laptop/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_empty_cup/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/shake_bottle/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/click_alarmclock/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/open_microwave/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_fan/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/shake_bottle_horizontally/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/click_bell/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/pick_diverse_bottles/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_mouse_pad/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/stack_blocks_three/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/dump_bin_bigbin/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/pick_dual_bottles/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_object_basket/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/stack_blocks_two/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/grab_roller/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_a2b_left/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_object_scale/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/stack_bowls_three/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/handover_block/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_a2b_right/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_object_stand/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/stack_bowls_two/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/handover_mic/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_bread_basket/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_phone_stand/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/stamp_seal/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/hanging_mug/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_bread_skillet/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_shoe/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/turn_switch/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/lift_pot/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/place_burger_fries/aloha-agilex_clean_50,/mnt/lijunhao/robotwin/press_stapler/aloha-agilex_clean_50"
output_dir=/mnt/lijunhao/value_function/outputs/v1
log_dir=${output_dir}/runs

lr=1e-5
value_head_lr=5e-5
batch_size=16
grad_accum_steps=8

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
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 10000 \
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
