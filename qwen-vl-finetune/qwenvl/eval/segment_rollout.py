from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Segment rollout value curves into rise/fall candidates.")
    parser.add_argument("--analysis_root", required=True, type=str)
    parser.add_argument("--input_manifest", default=None, type=str)
    parser.add_argument("--repo_id", default=None, type=str)
    parser.add_argument("--smooth_window", default=9, type=int)
    parser.add_argument("--rise_threshold", default=0.02, type=float)
    parser.add_argument("--fall_threshold", default=0.02, type=float)
    parser.add_argument("--min_segment_len", default=8, type=int)
    parser.add_argument("--min_segment_amplitude", default=0.08, type=float)
    parser.add_argument("--merge_gap", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


@dataclass
class Segment:
    start: int
    end: int
    trend: str

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _load_manifest(analysis_root: Path, manifest_path: Optional[str]) -> dict[str, Any]:
    path = Path(manifest_path).resolve() if manifest_path else analysis_root / "manifest.json"
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or len(values) <= 1:
        return values.astype(np.float32, copy=True)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def classify_trend_states(
    slope: np.ndarray,
    rise_threshold: float,
    fall_threshold: float,
) -> list[str]:
    rise_threshold = float(rise_threshold)
    fall_threshold = float(fall_threshold)
    release_rise = rise_threshold * 0.5
    release_fall = fall_threshold * 0.5
    state = "flat"
    states: list[str] = []
    for value in slope.tolist():
        if state == "flat":
            if value >= rise_threshold:
                state = "rise"
            elif value <= -fall_threshold:
                state = "fall"
        elif state == "rise":
            if value <= -fall_threshold:
                state = "fall"
            elif value < release_rise:
                state = "flat"
        elif state == "fall":
            if value >= rise_threshold:
                state = "rise"
            elif value > -release_fall:
                state = "flat"
        states.append(state)
    return states


def states_to_segments(states: list[str]) -> list[Segment]:
    if not states:
        return []
    segments: list[Segment] = []
    start = 0
    current = states[0]
    for idx in range(1, len(states)):
        if states[idx] != current:
            segments.append(Segment(start=start, end=idx - 1, trend=current))
            start = idx
            current = states[idx]
    segments.append(Segment(start=start, end=len(states) - 1, trend=current))
    return segments


def merge_same_trend_segments(segments: list[Segment], merge_gap: int) -> list[Segment]:
    if not segments:
        return []
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        new_segments: list[Segment] = []
        idx = 0
        while idx < len(merged):
            current = merged[idx]
            if (
                idx + 2 < len(merged)
                and merged[idx + 1].trend == "flat"
                and merged[idx + 1].length <= merge_gap
                and merged[idx + 2].trend == current.trend
                and current.trend in {"rise", "fall"}
            ):
                current = Segment(start=current.start, end=merged[idx + 2].end, trend=current.trend)
                idx += 3
                changed = True
            else:
                idx += 1
            new_segments.append(current)
        merged = new_segments
    return merged


def build_segment_records(
    rows: list[dict[str, Any]],
    smooth_window: int,
    rise_threshold: float,
    fall_threshold: float,
    min_segment_len: int,
    min_segment_amplitude: float,
    merge_gap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = np.asarray([float(row["absolute_value"]) for row in rows], dtype=np.float32)
    rel_values = np.asarray([float(row["relative_advantage"]) for row in rows], dtype=np.float32)
    smooth_values = moving_average(values, smooth_window)
    slope = np.diff(smooth_values, prepend=smooth_values[0])
    states = classify_trend_states(slope, rise_threshold=rise_threshold, fall_threshold=fall_threshold)
    segments = merge_same_trend_segments(states_to_segments(states), merge_gap=max(0, int(merge_gap)))

    segment_records: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        if segment.trend not in {"rise", "fall"}:
            continue
        amplitude = float(smooth_values[segment.end] - smooth_values[segment.start])
        if abs(amplitude) < float(min_segment_amplitude):
            continue
        if segment.length < int(min_segment_len):
            continue

        local_slice = slice(segment.start, segment.end + 1)
        local_values = smooth_values[local_slice]
        local_rel = rel_values[local_slice]
        if segment.trend == "rise":
            valley_offset = int(np.argmin(local_values))
            peak_offset = int(np.argmax(local_values))
        else:
            peak_offset = int(np.argmax(local_values))
            valley_offset = int(np.argmin(local_values))

        peak_idx = segment.start + peak_offset
        valley_idx = segment.start + valley_offset
        slope_mean = float(np.mean(slope[local_slice]))
        rel_mean = float(np.mean(np.abs(local_rel)))
        scale = max(float(min_segment_amplitude), 1e-6)
        trend_scale = max(float(min(rise_threshold, fall_threshold)), 1e-6)
        confidence = np.clip(
            0.45 * (abs(amplitude) / scale)
            + 0.30 * (abs(slope_mean) / trend_scale)
            + 0.25 * (rel_mean / trend_scale),
            0.0,
            1.0,
        )

        start_frame = int(rows[segment.start].get("dataset_frame_index", rows[segment.start]["frame_idx"]))
        end_frame = int(rows[segment.end].get("dataset_frame_index", rows[segment.end]["frame_idx"]))
        peak_frame = int(rows[peak_idx].get("dataset_frame_index", rows[peak_idx]["frame_idx"]))
        valley_frame = int(rows[valley_idx].get("dataset_frame_index", rows[valley_idx]["frame_idx"]))
        segment_records.append(
            {
                "segment_id": int(segment_id),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "trend": segment.trend,
                "amplitude": amplitude,
                "slope_mean": slope_mean,
                "peak_frame": peak_frame,
                "valley_frame": valley_frame,
                "confidence": float(confidence),
                "manual_label": None,
            }
        )

    summary = {
        "num_frames": int(len(rows)),
        "num_candidates": int(len(segment_records)),
        "state_counts": {
            "rise": int(sum(state == "rise" for state in states)),
            "fall": int(sum(state == "fall" for state in states)),
            "flat": int(sum(state == "flat" for state in states)),
        },
    }
    return segment_records, summary


def _write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    analysis_root = Path(args.analysis_root).resolve()
    manifest = _load_manifest(analysis_root, args.input_manifest)
    segments_root = analysis_root / "segments"
    summary_records = []

    for episode in manifest.get("episodes", []):
        metrics_rel = episode.get("metrics_jsonl")
        if not metrics_rel:
            continue
        metrics_path = analysis_root / metrics_rel
        episode_idx = int(episode["episode_idx"])
        stem = f"episode_{episode_idx:06d}"
        output_json = segments_root / f"{stem}.json"
        output_csv = segments_root / f"{stem}.csv"
        if output_json.exists() and output_csv.exists() and not args.overwrite:
            summary_records.append(
                {
                    "episode_idx": episode_idx,
                    "segments_json": str(output_json.relative_to(analysis_root)),
                    "segments_csv": str(output_csv.relative_to(analysis_root)),
                    "status": "skipped_existing",
                }
            )
            continue

        rows = load_metric_rows(metrics_path)
        segment_records, segment_summary = build_segment_records(
            rows=rows,
            smooth_window=args.smooth_window,
            rise_threshold=args.rise_threshold,
            fall_threshold=args.fall_threshold,
            min_segment_len=args.min_segment_len,
            min_segment_amplitude=args.min_segment_amplitude,
            merge_gap=args.merge_gap,
        )
        payload = {
            "episode_idx": episode_idx,
            "metrics_path": str(metrics_path.relative_to(analysis_root)),
            "repo_id": args.repo_id or manifest.get("repo_id"),
            "segments": segment_records,
            "summary": segment_summary,
        }
        _write_json(output_json, payload)
        _write_csv(output_csv, segment_records)
        summary_records.append(
            {
                "episode_idx": episode_idx,
                "segments_json": str(output_json.relative_to(analysis_root)),
                "segments_csv": str(output_csv.relative_to(analysis_root)),
                "num_candidates": int(len(segment_records)),
                "status": "ok",
            }
        )

    _write_json(
        analysis_root / "segmentation_summary.json",
        {
            "analysis_root": str(analysis_root),
            "repo_id": args.repo_id or manifest.get("repo_id"),
            "smooth_window": int(args.smooth_window),
            "rise_threshold": float(args.rise_threshold),
            "fall_threshold": float(args.fall_threshold),
            "min_segment_len": int(args.min_segment_len),
            "min_segment_amplitude": float(args.min_segment_amplitude),
            "merge_gap": int(args.merge_gap),
            "episodes": summary_records,
        },
    )


if __name__ == "__main__":
    main()
