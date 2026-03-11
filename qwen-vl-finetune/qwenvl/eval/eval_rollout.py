from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor_pair_map import parse_camera_names
from qwenvl.eval.value_evaluator import (
    build_rollout_evaluator,
    load_lerobot_metadata,
    load_run_config,
    resolve_checkpoint_path,
    resolve_parquet_path,
    resolve_prompt,
    resolve_video_paths,
)

logger = logging.getLogger(__name__)


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Unified rollout evaluation for pair and pi06 value functions.")
    parser.add_argument("--model_type", required=True, choices=["pair", "pi06"])
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--checkpoint_step", default="final_model", type=str)
    parser.add_argument("--repo_id", required=True, type=str)
    parser.add_argument("--analysis_root", default=None, type=str)
    parser.add_argument("--relative_interval", default=50, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--camera_names", default="cam_high,cam_left_wrist,cam_right_wrist", type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--frame_interval", default=1, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_episodes", default=None, type=int)
    parser.add_argument("--write_mode", default="both", choices=["parquet", "analysis", "both"])
    parser.add_argument("--save_plots", default=True, type=_str2bool)
    parser.add_argument("--attn_implementation", default="flash_attention_2", type=str)
    parser.add_argument("--bf16", default=True, type=_str2bool)
    return parser.parse_args(argv)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    return slug.strip("_") or "rollout"


def _resolve_analysis_root(args, repo_root: Path, checkpoint_path: Path) -> Path:
    if args.analysis_root:
        return Path(args.analysis_root).resolve()
    name = args.model_name or f"{args.model_type}_{checkpoint_path.parent.name}_{checkpoint_path.name}"
    return (repo_root / "rollout_analysis" / _safe_slug(name)).resolve()


def _analysis_enabled(write_mode: str) -> bool:
    return write_mode in {"analysis", "both"}


def _parquet_enabled(write_mode: str) -> bool:
    return write_mode in {"parquet", "both"}


def _read_frame_metadata(src_parquet: Path) -> list[int]:
    schema = pq.read_schema(src_parquet)
    if "frame_index" in schema.names:
        table = pq.read_table(src_parquet, columns=["frame_index"])
        frame_indices = [int(x) for x in table["frame_index"].to_pylist()]
    else:
        row_count = pq.read_metadata(src_parquet).num_rows
        frame_indices = list(range(int(row_count)))
    return frame_indices


def _attach_frame_metadata(rows: list[dict[str, Any]], frame_indices: list[int]) -> list[dict[str, Any]]:
    if len(rows) != len(frame_indices):
        raise ValueError(
            f"Prediction row count mismatch: predictions={len(rows)} frame_indices={len(frame_indices)}"
        )
    out = []
    for row in rows:
        frame_idx = int(row["frame_idx"])
        future_frame_idx = int(row["future_frame_idx"])
        enriched = dict(row)
        enriched["dataset_frame_index"] = int(frame_indices[frame_idx])
        enriched["dataset_future_frame_index"] = int(frame_indices[future_frame_idx])
        out.append(enriched)
    return out


def upsert_metric_columns(src_parquet: Path, output_path: Path, rows: list[dict[str, Any]]):
    table = pq.read_table(src_parquet)
    if len(rows) != table.num_rows:
        raise ValueError(
            f"Prediction row count mismatch for {src_parquet}: predictions={len(rows)}, parquet_rows={table.num_rows}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "relative_advantage": pa.array([float(row["relative_advantage"]) for row in rows], type=pa.float32()),
        "absolute_value": pa.array([float(row["absolute_value"]) for row in rows], type=pa.float32()),
        "absolute_advantage": pa.array([float(row["absolute_advantage"]) for row in rows], type=pa.float32()),
    }
    for name, value in columns.items():
        if name in table.column_names:
            index = table.column_names.index(name)
            table = table.set_column(index, name, value)
        else:
            table = table.append_column(name, value)
    pq.write_table(table, output_path)


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


def _save_plot(path: Path, rows: list[dict[str, Any]], title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [row["dataset_frame_index"] for row in rows]
    plt.figure(figsize=(12, 4.5))
    plt.plot(x_values, [row["absolute_value"] for row in rows], label="absolute_value", linewidth=1.6)
    plt.plot(x_values, [row["relative_advantage"] for row in rows], label="relative_advantage", linewidth=1.2)
    plt.plot(x_values, [row["absolute_advantage"] for row in rows], label="absolute_advantage", linewidth=1.2)
    plt.xlabel("frame_index")
    plt.ylabel("score")
    plt.ylim(-1.05, 1.05)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _episode_stem(episode_idx: int) -> str:
    return f"episode_{int(episode_idx):06d}"


def _build_summary(
    args,
    repo_root: Path,
    analysis_root: Path,
    checkpoint_path: Path,
    episodes: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "repo_id": str(repo_root),
        "analysis_root": str(analysis_root),
        "model_type": args.model_type,
        "model_name": args.model_name,
        "checkpoint_dir": str(checkpoint_path),
        "checkpoint_step": str(args.checkpoint_step),
        "relative_interval": int(args.relative_interval),
        "frame_interval": int(args.frame_interval),
        "write_mode": args.write_mode,
        "save_plots": bool(args.save_plots),
        "episodes": episodes,
        "run_config_snapshot": run_config,
    }


def main(argv: Optional[list[str]] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)
    repo_root = Path(args.repo_id).resolve()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_dir, args.checkpoint_step)
    run_config = load_run_config(checkpoint_path)
    base_model_name = run_config.get("model_args", {}).get("model_name_or_path")
    analysis_root = _resolve_analysis_root(args, repo_root, checkpoint_path)
    metrics_dir = analysis_root / "metrics"
    plots_dir = analysis_root / "plots"
    parquet_root = analysis_root / "parquet"

    if _parquet_enabled(args.write_mode) and int(args.frame_interval) != 1:
        raise ValueError("--frame_interval must be 1 when write_mode includes parquet to preserve row alignment.")

    metadata = load_lerobot_metadata(repo_root)
    total_episodes = int(getattr(metadata, "total_episodes", 0))
    if total_episodes <= 0:
        episodes = getattr(metadata, "episodes", None)
        if isinstance(episodes, dict):
            total_episodes = len(episodes)
        elif isinstance(episodes, list):
            total_episodes = len(episodes)
    if args.max_episodes is not None:
        total_episodes = min(total_episodes, int(args.max_episodes))

    camera_names = parse_camera_names(args.camera_names)
    evaluator = build_rollout_evaluator(
        model_type=args.model_type,
        checkpoint_dir=checkpoint_path,
        model_name_or_path=base_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attn_implementation=args.attn_implementation,
        bf16=args.bf16,
    )

    manifest_episodes: list[dict[str, Any]] = []
    try:
        for episode_idx in tqdm(range(total_episodes), desc=f"Evaluating {args.model_type} rollout"):
            src_parquet = resolve_parquet_path(repo_root, metadata, episode_idx)
            if not src_parquet.exists():
                logger.warning("Skipping episode %s because parquet is missing: %s", episode_idx, src_parquet)
                continue

            episode_stem = _episode_stem(episode_idx)
            metrics_jsonl_path = metrics_dir / f"{episode_stem}.jsonl"
            metrics_csv_path = metrics_dir / f"{episode_stem}.csv"
            plot_path = plots_dir / f"{episode_stem}.png"
            parquet_output_path = parquet_root / src_parquet.relative_to(repo_root / "data")

            outputs_exist = []
            if _analysis_enabled(args.write_mode):
                outputs_exist.extend([metrics_jsonl_path.exists(), metrics_csv_path.exists()])
                if args.save_plots:
                    outputs_exist.append(plot_path.exists())
            if _parquet_enabled(args.write_mode):
                outputs_exist.append(parquet_output_path.exists())
            if outputs_exist and all(outputs_exist) and not args.overwrite:
                manifest_episodes.append(
                    {
                        "episode_idx": int(episode_idx),
                        "metrics_jsonl": str(metrics_jsonl_path.relative_to(analysis_root)) if metrics_jsonl_path.exists() else None,
                        "metrics_csv": str(metrics_csv_path.relative_to(analysis_root)) if metrics_csv_path.exists() else None,
                        "plot": str(plot_path.relative_to(analysis_root)) if plot_path.exists() else None,
                        "parquet_output": str(parquet_output_path.relative_to(analysis_root)) if parquet_output_path.exists() else None,
                        "status": "skipped_existing",
                    }
                )
                continue

            prompt = resolve_prompt(metadata, episode_idx, args.prompt)
            video_paths = resolve_video_paths(repo_root, metadata, episode_idx, camera_names)
            frame_indices = _read_frame_metadata(src_parquet)
            sampled_frame_indices = frame_indices[:: max(1, int(args.frame_interval))]
            rows = evaluator.evaluate_video_metrics(
                video_paths=video_paths,
                prompt=prompt,
                relative_interval=args.relative_interval,
                frame_interval=args.frame_interval,
                min_frame_index=None,
                max_frame_index=None,
                prefetch=True,
                episode_idx=episode_idx,
            )
            rows = _attach_frame_metadata(rows, sampled_frame_indices)

            if _analysis_enabled(args.write_mode):
                _write_jsonl(metrics_jsonl_path, rows)
                _write_csv(metrics_csv_path, rows)
                if args.save_plots:
                    _save_plot(plot_path, rows, title=f"{args.model_type} value curves - episode {episode_idx}")

            if _parquet_enabled(args.write_mode):
                upsert_metric_columns(src_parquet=src_parquet, output_path=parquet_output_path, rows=rows)

            manifest_episodes.append(
                {
                    "episode_idx": int(episode_idx),
                    "num_frames": int(len(rows)),
                    "prompt": prompt,
                    "metrics_jsonl": str(metrics_jsonl_path.relative_to(analysis_root)) if _analysis_enabled(args.write_mode) else None,
                    "metrics_csv": str(metrics_csv_path.relative_to(analysis_root)) if _analysis_enabled(args.write_mode) else None,
                    "plot": str(plot_path.relative_to(analysis_root)) if _analysis_enabled(args.write_mode) and args.save_plots else None,
                    "parquet_output": str(parquet_output_path.relative_to(analysis_root)) if _parquet_enabled(args.write_mode) else None,
                    "status": "ok",
                }
            )
    finally:
        evaluator.shutdown()

    analysis_root.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(
        args=args,
        repo_root=repo_root,
        analysis_root=analysis_root,
        checkpoint_path=checkpoint_path,
        episodes=manifest_episodes,
        run_config=run_config,
    )
    manifest_path = analysis_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    with open(analysis_root / "run_summary.json", "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
