#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

model_type="pair"
model_name="pair_rollout"
checkpoint_dir="/path/to/checkpoint_dir"
checkpoint_step="final_model"
repo_id="/path/to/lerobot_dataset"
analysis_root=""
relative_interval=50
batch_size=8
camera_names="cam_high,cam_left_wrist,cam_right_wrist"
prompt=""
num_workers=8
frame_interval=1
max_episodes=""
write_mode="both"
save_plots="True"
attn_implementation="flash_attention_2"
bf16="True"
overwrite="False"

cmd=(
    python qwenvl/eval/eval_rollout.py
    --model_type "${model_type}"
    --model_name "${model_name}"
    --checkpoint_dir "${checkpoint_dir}"
    --checkpoint_step "${checkpoint_step}"
    --repo_id "${repo_id}"
    --relative_interval "${relative_interval}"
    --batch_size "${batch_size}"
    --camera_names "${camera_names}"
    --num_workers "${num_workers}"
    --frame_interval "${frame_interval}"
    --write_mode "${write_mode}"
    --save_plots "${save_plots}"
    --attn_implementation "${attn_implementation}"
    --bf16 "${bf16}"
)

if [ -n "${analysis_root}" ]; then
    cmd+=(--analysis_root "${analysis_root}")
fi

if [ -n "${prompt}" ]; then
    cmd+=(--prompt "${prompt}")
fi

if [ -n "${max_episodes}" ]; then
    cmd+=(--max_episodes "${max_episodes}")
fi

if [ "${overwrite}" = "True" ] || [ "${overwrite}" = "true" ]; then
    cmd+=(--overwrite)
fi

"${cmd[@]}"
