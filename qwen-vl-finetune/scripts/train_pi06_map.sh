#!/bin/bash
# export WANDB_API_KEY=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
# export WANDB_PROJECT="value-function"
# export WANDB_ENTITY="heisen0928-the-hong-kong-polytechnic-university"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=8

deepspeed=./scripts/zero3.json
entry_file=qwenvl/train/train_qwen_pi06.py

run_name="robotwin50_pi06_map"

llm=/mnt/lijunhao/output/value_function/checkpoints/Qwen2.5-VL-3B-Instruct-resize
datasets="/mnt/lijunhao/dataset/robotwin/move_can_pot/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_can_basket/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/put_bottles_dustbin/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/adjust_bottle/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/move_pillbottle_pad/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_cans_plasticbox/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/put_object_cabinet/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/beat_block_hammer/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/move_playingcard_away/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_container_plate/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/rotate_qrcode/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/blocks_ranking_rgb/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/move_stapler_pad/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_dual_shoes/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/scan_object/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/blocks_ranking_size/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/open_laptop/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_empty_cup/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/shake_bottle/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/click_alarmclock/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/open_microwave/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_fan/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/shake_bottle_horizontally/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/click_bell/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/pick_diverse_bottles/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_mouse_pad/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/stack_blocks_three/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/dump_bin_bigbin/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/pick_dual_bottles/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_object_basket/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/stack_blocks_two/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/grab_roller/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_a2b_left/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_object_scale/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/stack_bowls_three/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/handover_block/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_a2b_right/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_object_stand/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/stack_bowls_two/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/handover_mic/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_bread_basket/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_phone_stand/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/stamp_seal/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/hanging_mug/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_bread_skillet/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_shoe/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/turn_switch/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/lift_pot/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/place_burger_fries/aloha-agilex_clean_50,/mnt/lijunhao/dataset/robotwin/press_stapler/aloha-agilex_clean_50"
output_dir="/mnt/lijunhao/output/value_function/outputs/${run_name}"

lr=1e-5
batch_size=16
grad_accum_steps=8
max_steps=100_000
save_interval=5000
value_tokenizer_bins=201
value_tokenizer_min=-1.0
value_tokenizer_max=0.0

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --eval_strategy steps \
    --eval_steps 10_000 \
    --save_strategy steps \
    --save_steps ${save_interval} \
    --save_interval ${save_interval} \
    --save_total_limit 20 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --logging_first_step True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --use_value_tokenizer True \
    --value_tokenizer_bins ${value_tokenizer_bins} \
    --value_tokenizer_min ${value_tokenizer_min} \
    --value_tokenizer_max ${value_tokenizer_max} \
    --run_name ${run_name}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
