#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

analysis_root="/path/to/analysis_root"
input_manifest=""
repo_id=""
smooth_window=9
rise_threshold=0.02
fall_threshold=0.02
min_segment_len=8
min_segment_amplitude=0.08
merge_gap=4
overwrite="False"

if [ -z "${analysis_root}" ]; then
    echo "ANALYSIS_ROOT is required" >&2
    exit 1
fi

cmd=(
    python qwenvl/eval/segment_rollout.py
    --analysis_root "${analysis_root}"
    --smooth_window "${smooth_window}"
    --rise_threshold "${rise_threshold}"
    --fall_threshold "${fall_threshold}"
    --min_segment_len "${min_segment_len}"
    --min_segment_amplitude "${min_segment_amplitude}"
    --merge_gap "${merge_gap}"
)

if [ -n "${input_manifest}" ]; then
    cmd+=(--input_manifest "${input_manifest}")
fi

if [ -n "${repo_id}" ]; then
    cmd+=(--repo_id "${repo_id}")
fi

if [ "${overwrite}" = "True" ] || [ "${overwrite}" = "true" ]; then
    cmd+=(--overwrite)
fi

"${cmd[@]}"
