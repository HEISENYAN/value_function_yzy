from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.eval.segment_rollout import load_metric_rows
from qwenvl.eval.value_evaluator import resolve_parquet_path


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Build expert stage templates and reconstruct per-frame pseudo targets for rollout trajectories."
    )
    parser.add_argument("--expert_analysis_roots", required=True, type=str)
    parser.add_argument("--rollout_analysis_root", required=True, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--episode_outcomes_json", default=None, type=str)
    parser.add_argument("--export_dataset_root", default=None, type=str)
    parser.add_argument("--pseudo_value_column", default="pseudo_absolute_value", type=str)
    parser.add_argument("--rise_accept_ratio", default=0.5, type=float)
    parser.add_argument("--peak_separation_ratio", default=0.25, type=float)
    parser.add_argument("--failure_lambda", default=0.5, type=float)
    parser.add_argument("--success_max_drop_ratio", default=0.35, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _parse_analysis_roots(spec: str) -> list[Path]:
    roots = [Path(item).resolve() for item in str(spec).split(",") if item.strip()]
    if not roots:
        raise ValueError("--expert_analysis_roots is empty.")
    return roots


def _load_manifest(analysis_root: Path) -> dict[str, Any]:
    return _read_json(analysis_root / "manifest.json")


def _episode_stem(episode_idx: int) -> str:
    return f"episode_{int(episode_idx):06d}"


def _load_segment_payload(analysis_root: Path, episode_idx: int) -> dict[str, Any]:
    return _read_json(analysis_root / "segments" / f"{_episode_stem(episode_idx)}.json")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar-like value, got sequence={value}")
        return _safe_float(value[0])
    return float(value)


def _build_metric_index(rows: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], list[int]]:
    by_frame = {}
    ordered_frames = []
    for row in rows:
        frame = int(row.get("dataset_frame_index", row["frame_idx"]))
        by_frame[frame] = row
        ordered_frames.append(frame)
    return by_frame, ordered_frames


def _get_rise_segments(segment_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [segment for segment in segment_payload.get("segments", []) if segment.get("trend") == "rise"],
        key=lambda item: (int(item["start_frame"]), int(item["end_frame"])),
    )


def _get_fall_segments(segment_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [segment for segment in segment_payload.get("segments", []) if segment.get("trend") == "fall"],
        key=lambda item: (int(item["start_frame"]), int(item["end_frame"])),
    )


def _align_segments_to_stage_count(segments: list[dict[str, Any]], stage_count: int) -> list[dict[str, Any]]:
    if stage_count <= 0 or not segments:
        return []
    if stage_count == 1:
        return [max(segments, key=lambda item: (float(item.get("confidence", 0.0)), abs(float(item["amplitude"]))))]
    positions = np.linspace(0, len(segments) - 1, num=stage_count)
    return [segments[int(round(pos))] for pos in positions]


def build_task_templates(expert_analysis_roots: list[Path]) -> dict[str, Any]:
    task_samples: dict[str, list[dict[str, Any]]] = {}
    for analysis_root in expert_analysis_roots:
        manifest = _load_manifest(analysis_root)
        for episode in manifest.get("episodes", []):
            if episode.get("status") not in {None, "ok", "skipped_existing"}:
                continue
            prompt = episode.get("prompt")
            if not prompt or not episode.get("metrics_jsonl"):
                continue
            episode_idx = int(episode["episode_idx"])
            segment_path = analysis_root / "segments" / f"{_episode_stem(episode_idx)}.json"
            if not segment_path.exists():
                continue
            segments = _load_segment_payload(analysis_root, episode_idx)
            rises = _get_rise_segments(segments)
            if not rises:
                continue
            task_samples.setdefault(prompt, []).append(
                {
                    "analysis_root": str(analysis_root),
                    "episode_idx": episode_idx,
                    "rises": rises,
                    "num_frames": int(segments.get("summary", {}).get("num_frames", 0)),
                }
            )

    templates: dict[str, Any] = {}
    for prompt, samples in task_samples.items():
        rise_counts = [len(sample["rises"]) for sample in samples if sample["rises"]]
        if not rise_counts:
            continue
        stage_count = max(1, int(round(float(np.median(rise_counts)))))
        stage_anchor_values = [float(stage_idx / stage_count) for stage_idx in range(stage_count + 1)]
        stage_stats: list[dict[str, Any]] = []
        for stage_idx in range(stage_count):
            aligned_segments = []
            for sample in samples:
                aligned = _align_segments_to_stage_count(sample["rises"], stage_count)
                if aligned:
                    aligned_segments.append(aligned[stage_idx])
            amplitudes = [abs(float(seg["amplitude"])) for seg in aligned_segments]
            lengths = [int(seg["end_frame"]) - int(seg["start_frame"]) + 1 for seg in aligned_segments]
            confidences = [float(seg.get("confidence", 0.0)) for seg in aligned_segments]
            amp_median = float(np.median(amplitudes)) if amplitudes else 1e-3
            stage_stats.append(
                {
                    "stage_idx": int(stage_idx + 1),
                    "anchor_value": float(stage_anchor_values[stage_idx + 1]),
                    "amp_median": max(amp_median, 1e-3),
                    "amp_p25": float(np.percentile(amplitudes, 25)) if amplitudes else amp_median,
                    "amp_p75": float(np.percentile(amplitudes, 75)) if amplitudes else amp_median,
                    "length_median": int(round(float(np.median(lengths)))) if lengths else 1,
                    "confidence_median": float(np.median(confidences)) if confidences else 0.0,
                    "support": int(len(aligned_segments)),
                }
            )
        templates[prompt] = {
            "prompt": prompt,
            "stage_count": int(stage_count),
            "anchor_values": stage_anchor_values,
            "stages": stage_stats,
            "support": int(len(samples)),
        }
    return templates


def _load_episode_outcomes(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    payload = _read_json(Path(path).resolve())
    if isinstance(payload, dict) and "episodes" in payload and isinstance(payload["episodes"], dict):
        return payload["episodes"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("episode_outcomes_json must be an object mapping episode ids to outcome payloads.")


def _find_following_fall(
    fall_segments: list[dict[str, Any]],
    peak_frame: int,
    next_peak_frame: Optional[int],
) -> Optional[dict[str, Any]]:
    best = None
    for segment in fall_segments:
        start = int(segment["start_frame"])
        if start <= peak_frame:
            continue
        if next_peak_frame is not None and start >= next_peak_frame:
            break
        if best is None or int(segment["valley_frame"]) > int(best["valley_frame"]):
            best = segment
    return best


def _add_or_update_anchor(anchor_map: dict[int, dict[str, Any]], frame: int, value: float, kind: str, stage_idx: int):
    frame = int(frame)
    value = _clip01(value)
    previous = anchor_map.get(frame)
    if previous is None or value > float(previous["value"]) or kind in {"final_success", "final_failure"}:
        anchor_map[frame] = {
            "frame": frame,
            "value": value,
            "kind": kind,
            "stage_idx": int(stage_idx),
        }


def _interpolate_targets(
    ordered_frames: list[int],
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame_to_pos = {int(frame): idx for idx, frame in enumerate(ordered_frames)}
    sorted_anchors = []
    for anchor in anchors:
        frame = int(anchor["frame"])
        if frame in frame_to_pos:
            item = dict(anchor)
            item["position"] = int(frame_to_pos[frame])
            sorted_anchors.append(item)
    sorted_anchors.sort(key=lambda item: item["position"])
    if not sorted_anchors:
        return [
            {
                "frame": int(frame),
                "position": int(pos),
                "pseudo_absolute_value": 0.0,
            }
            for pos, frame in enumerate(ordered_frames)
        ]

    values = np.zeros(len(ordered_frames), dtype=np.float32)
    first = sorted_anchors[0]
    values[: first["position"] + 1] = float(first["value"])
    for left, right in zip(sorted_anchors[:-1], sorted_anchors[1:]):
        left_pos = int(left["position"])
        right_pos = int(right["position"])
        left_val = float(left["value"])
        right_val = float(right["value"])
        if right_pos <= left_pos:
            values[left_pos] = right_val
            continue
        values[left_pos : right_pos + 1] = np.linspace(left_val, right_val, num=(right_pos - left_pos + 1), dtype=np.float32)
    last = sorted_anchors[-1]
    values[last["position"] :] = float(last["value"])

    return [
        {
            "frame": int(frame),
            "position": int(pos),
            "pseudo_absolute_value": float(values[pos]),
        }
        for pos, frame in enumerate(ordered_frames)
    ]


def reconstruct_episode_targets(
    rows: list[dict[str, Any]],
    segment_payload: dict[str, Any],
    template: dict[str, Any],
    outcome_override: Optional[dict[str, Any]],
    rise_accept_ratio: float,
    peak_separation_ratio: float,
    failure_lambda: float,
    success_max_drop_ratio: float,
) -> dict[str, Any]:
    metrics_by_frame, ordered_frames = _build_metric_index(rows)
    rises = _get_rise_segments(segment_payload)
    falls = _get_fall_segments(segment_payload)
    stage_count = int(template["stage_count"])
    anchor_values = [float(x) for x in template["anchor_values"]]
    stages = list(template["stages"])
    eps = 1e-6

    accepted: list[dict[str, Any]] = []
    partial_best: Optional[dict[str, Any]] = None
    current_stage = 1
    last_accepted_peak_raw: Optional[float] = None

    for rise in rises:
        if current_stage > stage_count:
            break
        stage_ref = max(float(stages[current_stage - 1]["amp_median"]), eps)
        peak_frame = int(rise["peak_frame"])
        valley_frame = int(rise["valley_frame"])
        peak_raw = float(metrics_by_frame[peak_frame]["absolute_value"])
        valley_raw = float(metrics_by_frame[valley_frame]["absolute_value"])
        rise_gain = max(0.0, peak_raw - valley_raw)
        separated = last_accepted_peak_raw is None or peak_raw >= last_accepted_peak_raw + peak_separation_ratio * stage_ref
        accepted_now = rise_gain >= rise_accept_ratio * stage_ref and separated
        candidate = {
            "stage_idx": int(current_stage),
            "segment": rise,
            "peak_frame": peak_frame,
            "valley_frame": valley_frame,
            "peak_raw": peak_raw,
            "valley_raw": valley_raw,
            "rise_gain": rise_gain,
        }
        if accepted_now:
            accepted.append(candidate)
            last_accepted_peak_raw = peak_raw
            current_stage += 1
        else:
            if partial_best is None or peak_raw > float(partial_best["peak_raw"]):
                partial_best = candidate

    accepted_count = len(accepted)
    final_frame = int(ordered_frames[-1])
    final_raw = float(metrics_by_frame[final_frame]["absolute_value"])

    inferred_success = False
    if accepted_count >= stage_count and accepted:
        last_stage_ref = max(float(stages[-1]["amp_median"]), eps)
        last_peak_raw = float(accepted[-1]["peak_raw"])
        drop_ratio = max(0.0, last_peak_raw - final_raw) / last_stage_ref
        inferred_success = drop_ratio <= float(success_max_drop_ratio)

    success = inferred_success
    terminal_anchor_override = None
    if outcome_override:
        if "success" in outcome_override:
            success = bool(outcome_override["success"])
        if "terminal_anchor" in outcome_override:
            terminal_anchor_override = float(outcome_override["terminal_anchor"])

    anchor_map: dict[int, dict[str, Any]] = {}
    first_frame = int(ordered_frames[0])
    _add_or_update_anchor(anchor_map, first_frame, 0.0, "start", 0)

    anchor_debug: list[dict[str, Any]] = []
    for idx, accepted_item in enumerate(accepted):
        stage_idx = int(accepted_item["stage_idx"])
        peak_frame = int(accepted_item["peak_frame"])
        peak_value = float(anchor_values[stage_idx])
        _add_or_update_anchor(anchor_map, peak_frame, peak_value, "stage_peak", stage_idx)
        anchor_debug.append({"frame": peak_frame, "value": peak_value, "kind": "stage_peak", "stage_idx": stage_idx})

        next_peak_frame = int(accepted[idx + 1]["peak_frame"]) if idx + 1 < len(accepted) else None
        fall_segment = _find_following_fall(falls, peak_frame=peak_frame, next_peak_frame=next_peak_frame)
        if fall_segment is not None:
            valley_frame = int(fall_segment["valley_frame"])
            valley_raw = float(metrics_by_frame[valley_frame]["absolute_value"])
            peak_raw = float(accepted_item["peak_raw"])
            stage_ref = max(float(stages[stage_idx - 1]["amp_median"]), eps)
            drop_ratio = np.clip((peak_raw - valley_raw) / stage_ref, 0.0, 1.0)
            prev_anchor = float(anchor_values[stage_idx - 1]) if stage_idx > 0 else 0.0
            stage_anchor = float(anchor_values[stage_idx])
            valley_value = max(prev_anchor, stage_anchor - float(drop_ratio) * (stage_anchor - prev_anchor))
            _add_or_update_anchor(anchor_map, valley_frame, valley_value, "stage_valley", stage_idx)
            anchor_debug.append(
                {"frame": valley_frame, "value": valley_value, "kind": "stage_valley", "stage_idx": stage_idx}
            )

    if success:
        terminal_value = 1.0 if terminal_anchor_override is None else _clip01(float(terminal_anchor_override))
        _add_or_update_anchor(anchor_map, final_frame, terminal_value, "final_success", stage_count)
        anchor_debug.append(
            {"frame": final_frame, "value": terminal_value, "kind": "final_success", "stage_idx": stage_count}
        )
        partial_summary = None
    else:
        if accepted_count < stage_count:
            next_stage = accepted_count + 1
            current_anchor = float(anchor_values[accepted_count])
            next_anchor = float(anchor_values[next_stage])
            stage_span = next_anchor - current_anchor
            stage_ref = max(float(stages[next_stage - 1]["amp_median"]), eps)
            if partial_best is not None:
                partial_progress = np.clip((float(partial_best["peak_raw"]) - float(partial_best["valley_raw"])) / stage_ref, 0.0, 1.0)
                partial_peak_value = current_anchor + float(partial_progress) * stage_span
                _add_or_update_anchor(
                    anchor_map,
                    int(partial_best["peak_frame"]),
                    partial_peak_value,
                    "partial_peak",
                    next_stage,
                )
                rollback = np.clip((float(partial_best["peak_raw"]) - final_raw) / stage_ref, 0.0, 1.0)
                terminal_value = np.clip(
                    partial_peak_value - float(failure_lambda) * float(rollback) * stage_span,
                    current_anchor,
                    next_anchor - eps,
                )
                partial_summary = {
                    "stage_idx": int(next_stage),
                    "peak_frame": int(partial_best["peak_frame"]),
                    "peak_value": float(partial_peak_value),
                    "progress_ratio": float(partial_progress),
                    "rollback_ratio": float(rollback),
                }
            else:
                terminal_value = current_anchor
                partial_summary = None
            terminal_stage_idx = next_stage
        else:
            current_anchor = float(anchor_values[-2]) if stage_count > 1 else 0.0
            stage_ref = max(float(stages[-1]["amp_median"]), eps)
            stage_span = float(anchor_values[-1] - current_anchor)
            best_peak_raw = float(accepted[-1]["peak_raw"]) if accepted else final_raw
            rollback = np.clip((best_peak_raw - final_raw) / stage_ref, 0.0, 1.0)
            terminal_value = np.clip(
                1.0 - float(failure_lambda) * float(rollback) * stage_span,
                current_anchor,
                1.0 - eps,
            )
            partial_summary = {
                "stage_idx": int(stage_count),
                "peak_frame": int(accepted[-1]["peak_frame"]) if accepted else final_frame,
                "peak_value": 1.0,
                "progress_ratio": 1.0,
                "rollback_ratio": float(rollback),
            }
            terminal_stage_idx = stage_count

        if terminal_anchor_override is not None:
            upper = 1.0 - eps if accepted_count >= stage_count else float(anchor_values[min(accepted_count + 1, stage_count)]) - eps
            lower = float(anchor_values[min(accepted_count, stage_count)])
            terminal_value = float(np.clip(float(terminal_anchor_override), lower, max(lower, upper)))
        _add_or_update_anchor(anchor_map, final_frame, terminal_value, "final_failure", terminal_stage_idx)
        anchor_debug.append(
            {"frame": final_frame, "value": float(terminal_value), "kind": "final_failure", "stage_idx": terminal_stage_idx}
        )

    anchors = sorted(anchor_map.values(), key=lambda item: int(item["frame"]))
    dense_targets = _interpolate_targets(ordered_frames, anchors)
    dense_by_frame = {int(item["frame"]): item for item in dense_targets}

    pseudo_rows = []
    for row in rows:
        frame = int(row.get("dataset_frame_index", row["frame_idx"]))
        pseudo_value = float(dense_by_frame[frame]["pseudo_absolute_value"])
        pseudo_rows.append(
            {
                **row,
                "pseudo_absolute_value": pseudo_value,
            }
        )

    return {
        "pseudo_rows": pseudo_rows,
        "anchors": anchors,
        "accepted_stages": [
            {
                "stage_idx": int(item["stage_idx"]),
                "peak_frame": int(item["peak_frame"]),
                "peak_raw": float(item["peak_raw"]),
                "rise_gain": float(item["rise_gain"]),
            }
            for item in accepted
        ],
        "partial_summary": partial_summary,
        "success": bool(success),
        "inferred_success": bool(inferred_success),
        "completed_stage_count": int(accepted_count),
        "stage_count": int(stage_count),
        "terminal_value": float(pseudo_rows[-1]["pseudo_absolute_value"]),
        "anchor_debug": anchor_debug,
    }


def _symlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        if src.is_dir():
            os.symlink(src.resolve(), dst, target_is_directory=True)
        else:
            os.symlink(src.resolve(), dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def _prepare_export_dataset_root(source_repo_root: Path, export_dataset_root: Path):
    export_dataset_root.mkdir(parents=True, exist_ok=True)
    for child in source_repo_root.iterdir():
        if child.name == "data":
            continue
        _symlink_or_copy(child, export_dataset_root / child.name)
    (export_dataset_root / "data").mkdir(parents=True, exist_ok=True)


def _write_pseudo_parquet(
    src_parquet: Path,
    dst_parquet: Path,
    pseudo_rows: list[dict[str, Any]],
    column_name: str,
):
    table = pq.read_table(src_parquet)
    if len(pseudo_rows) != table.num_rows:
        raise ValueError(
            f"Pseudo row count mismatch for {src_parquet}: pseudo_rows={len(pseudo_rows)}, parquet_rows={table.num_rows}"
        )
    values = pa.array([float(row[column_name]) for row in pseudo_rows], type=pa.float32())
    if column_name in table.column_names:
        index = table.column_names.index(column_name)
        table = table.set_column(index, column_name, values)
    else:
        table = table.append_column(column_name, values)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_parquet)


def _resolve_template_for_prompt(prompt: str, templates: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if prompt in templates:
        return prompt, templates[prompt]
    if len(templates) == 1:
        only_key = next(iter(templates))
        return only_key, templates[only_key]
    raise KeyError(f"Could not resolve template for prompt={prompt!r}. Available templates={list(templates.keys())}")


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    expert_roots = _parse_analysis_roots(args.expert_analysis_roots)
    rollout_analysis_root = Path(args.rollout_analysis_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else rollout_analysis_root / "pseudo_targets"
    output_root.mkdir(parents=True, exist_ok=True)

    rollout_manifest = _load_manifest(rollout_analysis_root)
    if int(rollout_manifest.get("frame_interval", 1)) != 1:
        raise ValueError("rollout analysis manifest must have frame_interval=1 for pseudo target reconstruction.")

    templates = build_task_templates(expert_roots)
    if not templates:
        raise ValueError("No expert stage templates could be built from --expert_analysis_roots.")
    _write_json(output_root / "stage_templates.json", {"templates": templates})

    outcome_overrides = _load_episode_outcomes(args.episode_outcomes_json)
    export_dataset_root = Path(args.export_dataset_root).resolve() if args.export_dataset_root else None
    rollout_repo_root = Path(rollout_manifest["repo_id"]).resolve()
    rollout_metadata = None
    if export_dataset_root is not None:
        _prepare_export_dataset_root(rollout_repo_root, export_dataset_root)
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        rollout_metadata = LeRobotDatasetMetadata(str(rollout_repo_root))

    episode_summaries = []
    for episode in rollout_manifest.get("episodes", []):
        if episode.get("status") not in {None, "ok", "skipped_existing"}:
            continue
        prompt = episode.get("prompt")
        metrics_rel = episode.get("metrics_jsonl")
        if not prompt or not metrics_rel:
            continue
        episode_idx = int(episode["episode_idx"])
        episode_output_json = output_root / "episodes" / f"{_episode_stem(episode_idx)}.json"
        episode_output_jsonl = output_root / "episodes" / f"{_episode_stem(episode_idx)}.jsonl"
        episode_output_csv = output_root / "episodes" / f"{_episode_stem(episode_idx)}.csv"
        if (
            episode_output_json.exists()
            and episode_output_jsonl.exists()
            and episode_output_csv.exists()
            and not args.overwrite
        ):
            episode_summaries.append(
                {
                    "episode_idx": int(episode_idx),
                    "status": "skipped_existing",
                    "pseudo_metrics_jsonl": str(episode_output_jsonl.relative_to(output_root)),
                }
            )
            continue

        _, template = _resolve_template_for_prompt(prompt, templates)
        metrics_rows = load_metric_rows(rollout_analysis_root / metrics_rel)
        segment_payload = _load_segment_payload(rollout_analysis_root, episode_idx)
        outcome_override = outcome_overrides.get(str(episode_idx), outcome_overrides.get(int(episode_idx), None))

        reconstruction = reconstruct_episode_targets(
            rows=metrics_rows,
            segment_payload=segment_payload,
            template=template,
            outcome_override=outcome_override,
            rise_accept_ratio=args.rise_accept_ratio,
            peak_separation_ratio=args.peak_separation_ratio,
            failure_lambda=args.failure_lambda,
            success_max_drop_ratio=args.success_max_drop_ratio,
        )

        _write_jsonl(episode_output_jsonl, reconstruction["pseudo_rows"])
        _write_csv(episode_output_csv, reconstruction["pseudo_rows"])
        _write_json(
            episode_output_json,
            {
                "episode_idx": int(episode_idx),
                "prompt": prompt,
                "success": bool(reconstruction["success"]),
                "inferred_success": bool(reconstruction["inferred_success"]),
                "completed_stage_count": int(reconstruction["completed_stage_count"]),
                "stage_count": int(reconstruction["stage_count"]),
                "terminal_value": float(reconstruction["terminal_value"]),
                "accepted_stages": reconstruction["accepted_stages"],
                "partial_summary": reconstruction["partial_summary"],
                "anchors": reconstruction["anchors"],
            },
        )

        parquet_rel = None
        if export_dataset_root is not None:
            if rollout_metadata is None:
                raise RuntimeError("rollout metadata is not initialized for dataset export.")
            src_parquet = resolve_parquet_path(rollout_repo_root, rollout_metadata, episode_idx)
            dst_parquet = export_dataset_root / src_parquet.relative_to(rollout_repo_root)
            _write_pseudo_parquet(
                src_parquet=src_parquet,
                dst_parquet=dst_parquet,
                pseudo_rows=reconstruction["pseudo_rows"],
                column_name=args.pseudo_value_column,
            )
            parquet_rel = str(dst_parquet.relative_to(export_dataset_root))

        episode_summaries.append(
            {
                "episode_idx": int(episode_idx),
                "prompt": prompt,
                "success": bool(reconstruction["success"]),
                "completed_stage_count": int(reconstruction["completed_stage_count"]),
                "stage_count": int(reconstruction["stage_count"]),
                "terminal_value": float(reconstruction["terminal_value"]),
                "pseudo_metrics_jsonl": str(episode_output_jsonl.relative_to(output_root)),
                "pseudo_metrics_csv": str(episode_output_csv.relative_to(output_root)),
                "episode_summary_json": str(episode_output_json.relative_to(output_root)),
                "exported_parquet": parquet_rel,
                "status": "ok",
            }
        )

    summary = {
        "expert_analysis_roots": [str(path) for path in expert_roots],
        "rollout_analysis_root": str(rollout_analysis_root),
        "output_root": str(output_root),
        "export_dataset_root": str(export_dataset_root) if export_dataset_root is not None else None,
        "pseudo_value_column": args.pseudo_value_column,
        "episodes": episode_summaries,
    }
    _write_json(output_root / "reconstruction_summary.json", summary)


if __name__ == "__main__":
    main()
