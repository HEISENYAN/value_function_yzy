#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

expert_analysis_roots="/path/to/expert_analysis_root"
rollout_analysis_root="/path/to/rollout_analysis_root"
output_root=""
episode_outcomes_json=""
export_dataset_root="/path/to/pseudo_dataset_root"
pseudo_value_column="pseudo_absolute_value"
rise_accept_ratio=0.5
peak_separation_ratio=0.25
failure_lambda=0.5
success_max_drop_ratio=0.35
overwrite="False"

cmd=(
    python qwenvl/eval/reconstruct_rollout_targets.py
    --expert_analysis_roots "${expert_analysis_roots}"
    --rollout_analysis_root "${rollout_analysis_root}"
    --export_dataset_root "${export_dataset_root}"
    --pseudo_value_column "${pseudo_value_column}"
    --rise_accept_ratio "${rise_accept_ratio}"
    --peak_separation_ratio "${peak_separation_ratio}"
    --failure_lambda "${failure_lambda}"
    --success_max_drop_ratio "${success_max_drop_ratio}"
)

if [ -n "${output_root}" ]; then
    cmd+=(--output_root "${output_root}")
fi

if [ -n "${episode_outcomes_json}" ]; then
    cmd+=(--episode_outcomes_json "${episode_outcomes_json}")
fi

if [ "${overwrite}" = "True" ] || [ "${overwrite}" = "true" ]; then
    cmd+=(--overwrite)
fi

"${cmd[@]}"

