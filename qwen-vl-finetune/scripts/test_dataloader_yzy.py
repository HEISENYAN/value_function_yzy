#!/usr/bin/env python3
"""
测试 data_loader_yzy.py 在多 worker DataLoader 下是否能完整跑完 episode。

支持两种模式：
1) raw: 直接读取 LeRobotValueDataset（输出包含 PIL Image / meta_ep_idx / meta_R），并校验每个 episode 的输出帧数是否等于 metadata 里的 length。
2) trainer: 尽量复刻 train_qwen.py 的 trainer 读取路径：
   LeRobotValueDataset -> IterableSupervisedDataset(预处理/打 token/算 position_ids) -> DataCollator -> DataLoader

示例：
  # raw：多 worker 完整跑完一个 split，并校验每个 episode 是否完整
  python scripts/test_dataloader_yzy.py --mode raw --dataset_dir /path/to/lerobot --num_workers 8 --buffer_size 1

  # trainer：复刻 Trainer 的 batch 构建（需要 model/processor）
  python scripts/test_dataloader_yzy.py --mode trainer --dataset_dir /path/to/lerobot --model_name_or_path /path/to/Qwen2.5-VL-3B-Instruct-resize --num_workers 8 --batch_size 2 --max_batches 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


def _repo_root() -> Path:
    # scripts/.. -> qwen-vl-finetune/
    return Path(__file__).resolve().parent.parent


def _setup_sys_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.append(str(root))


def _parse_camera_names(s: str) -> List[str]:
    # "cam_high,cam_left_wrist,cam_right_wrist"
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _resolve_dataset_dir(p: str) -> str:
    # 与 make_supervised_data_module 里的逻辑保持一致：优先原路径，否则相对 repo_root 再拼一次
    if os.path.exists(p):
        return p
    cand = os.path.abspath(os.path.join(str(_repo_root()), p))
    return cand


def _collate_raw(batch: List[Dict[str, Any]]) -> Any:
    # LeRobotValueDataset 里包含 PIL Image，默认 collate 会失败；这里直接透传
    if len(batch) == 1:
        return batch[0]
    return batch


@dataclass
class _MinimalDataArgs:
    # 仅保留 data_processor.IterableSupervisedDataset 用到的字段
    model_type: str = "qwen2.5vl"
    data_packing: bool = False
    data_flatten: bool = False

    # processor pixels
    min_pixels: int = 28 * 28 * 16
    max_pixels: int = 28 * 28 * 576
    video_min_pixels: int = 256 * 28 * 28
    video_max_pixels: int = 1024 * 28 * 28
    video_min_frames: int = 4
    video_max_frames: int = 8
    video_fps: float = 2.0

    # 为了和训练参数一致（虽然这里不直接使用）
    use_value_tokenizer: bool = False
    value_tokenizer_bins: int = 201
    value_tokenizer_min: float = -1.0
    value_tokenizer_max: float = 0.0

    # make_supervised_data_module 的入参名里叫 dataset_use，这里不一定需要，但保留
    dataset_use: str = ""
    val_ratio: float = 0.1
    seed: int = 42
    buffer_size: int = 5000
    camera_names: Optional[List[str]] = None


def run_raw(args: argparse.Namespace) -> int:
    from qwenvl.data.data_loader_yzy import LeRobotValueDataset

    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {args.dataset_dir} -> {dataset_dir}")

    ds = LeRobotValueDataset(
        dataset_dir=dataset_dir,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        buffer_size=args.buffer_size,
        camera_names=_parse_camera_names(args.camera_names),
        max_episodes=args.max_episodes,
    )

    expected_len_by_ep = {int(e["episode_idx"]): int(e["length"]) for e in ds.episodes_meta}
    expected_total = int(len(ds))

    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=_collate_raw,
    )

    t0 = time.time()
    counts = defaultdict(int)
    total = 0
    bad_samples = 0

    # 这里会真正触发 multi-worker + pyav 解码读取
    for item in dl:
        try:
            ep = int(item.get("meta_ep_idx", -1))
            counts[ep] += 1
            total += 1
            if args.print_every > 0 and total % args.print_every == 0:
                dt = time.time() - t0
                rate = total / max(dt, 1e-9)
                print(f"[raw] progress: {total}/{expected_total} samples, {rate:.2f} samples/s")
        except Exception:
            bad_samples += 1
            traceback.print_exc()
            if args.fail_fast:
                raise

        if args.max_samples is not None and total >= args.max_samples:
            break

    # 校验 episode 完整性：如果 data_loader_yzy 某个 episode 里抛异常，会被 catch 后 continue，
    # 这样该 episode 的 frames 会整体缺失 -> counts 会小于 expected。
    missing_eps = []
    for ep, exp in sorted(expected_len_by_ep.items()):
        got = int(counts.get(ep, 0))
        if got != exp:
            missing_eps.append((ep, got, exp))

    dt = time.time() - t0
    print(f"[raw] done. total_yielded={total}, expected_total={expected_total}, bad_samples={bad_samples}, time={dt:.2f}s")

    if missing_eps:
        print(f"[raw] ERROR: {len(missing_eps)} episode(s) incomplete:")
        for ep, got, exp in missing_eps[:50]:
            print(f"  - episode_idx={ep}: got={got}, expected={exp}")
        if len(missing_eps) > 50:
            print(f"  ... and {len(missing_eps) - 50} more")
        return 2

    if args.max_samples is None and total != expected_total:
        print(f"[raw] ERROR: total samples mismatch: got={total}, expected={expected_total}")
        return 3

    print("[raw] OK: all episodes complete (counts match metadata).")
    return 0


def run_trainer_like(args: argparse.Namespace) -> int:
    """
    复刻 Trainer 读取路径：
      LeRobotValueDataset -> IterableSupervisedDataset -> DataCollator -> DataLoader
    这里主要验证：多 worker + 预处理/打 token/算 position_ids + collate 是否会炸。
    """
    from transformers import AutoProcessor
    import transformers

    from qwenvl.data.data_loader_yzy import LeRobotValueDataset
    from qwenvl.data.data_processor import IterableSupervisedDataset, DataCollatorForSupervisedDataset

    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {args.dataset_dir} -> {dataset_dir}")

    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required in trainer mode")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # 训练里 Trainer 传 processing_class=tokenizer，但 dataset 内部用 processor.tokenizer。
    # 这里把 collator/tokenizer 与 processor 对齐。
    processor.tokenizer = tokenizer

    data_args = _MinimalDataArgs(
        model_type="qwen2.5vl",
        data_packing=False,
        data_flatten=False,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_min_frames=args.video_min_frames,
        video_max_frames=args.video_max_frames,
        video_fps=args.video_fps,
        dataset_use=dataset_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        buffer_size=args.buffer_size,
        camera_names=_parse_camera_names(args.camera_names),
    )

    le_ds = LeRobotValueDataset(
        dataset_dir=dataset_dir,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        buffer_size=args.buffer_size,
        camera_names=_parse_camera_names(args.camera_names),
        max_episodes=args.max_episodes,
    )
    expected_samples = int(len(le_ds))
    ds = IterableSupervisedDataset(le_ds, processor, data_args, value_tokenizer=None)
    collator = DataCollatorForSupervisedDataset(tokenizer)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=bool(args.num_workers > 0) if args.persistent_workers else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collator,
        drop_last=args.drop_last,
    )

    max_batches = None if args.full_epoch else args.max_batches
    epochs = int(args.epochs)

    t0 = time.time()
    total_batches = 0
    total_samples = 0

    # 一个 epoch：把 iterable DataLoader 消耗到自然结束
    for ep in range(epochs):
        ep_t0 = time.time()
        ep_batches = 0
        ep_samples = 0

        for batch in dl:
            ep_batches += 1
            total_batches += 1

            # 粗略统计样本数（batch_size 维度）
            try:
                bsz = int(batch["input_ids"].shape[0])
            except Exception:
                bsz = args.batch_size
            ep_samples += bsz
            total_samples += bsz

            if args.print_every > 0 and total_batches % args.print_every == 0:
                dt = time.time() - t0
                rate_b = total_batches / max(dt, 1e-9)
                rate_s = total_samples / max(dt, 1e-9)
                print(
                    f"[trainer] epoch={ep+1}/{epochs} batches={total_batches} samples={total_samples} "
                    f"({rate_b:.2f} batch/s, {rate_s:.2f} sample/s)"
                )

            if max_batches is not None and total_batches >= max_batches:
                break

        ep_dt = time.time() - ep_t0
        print(f"[trainer] epoch {ep+1}/{epochs} done. epoch_batches={ep_batches}, epoch_samples={ep_samples}, time={ep_dt:.2f}s")

        if max_batches is not None and total_batches >= max_batches:
            break

    dt = time.time() - t0
    print(f"[trainer] done. batches={total_batches}, samples={total_samples}, time={dt:.2f}s")

    if args.strict_expected and args.full_epoch and epochs == 1 and not args.drop_last:
        if total_samples != expected_samples:
            print(f"[trainer] ERROR: sample count mismatch: got={total_samples}, expected={expected_samples}")
            return 4
        print("[trainer] OK: sample count matches LeRobotValueDataset.__len__().")
    return 0


def main() -> int:
    _setup_sys_path()

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["raw", "trainer"], default="raw")
    p.add_argument("--dataset_dir", required=True, help="LeRobot dataset directory")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer_size", type=int, default=1, help="建议 raw 测试用 1，尽量减少 shuffle 影响")
    p.add_argument("--camera_names", type=str, default="cam_high,cam_left_wrist,cam_right_wrist")
    p.add_argument("--max_episodes", type=int, default=None)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--fail_fast", action="store_true")
    p.add_argument(
        "--mp_start_method",
        type=str,
        default=None,
        choices=[None, "fork", "spawn", "forkserver"],
        help="多进程启动方式；遇到 pyav/fork 不兼容或 hang 时优先试 --mp_start_method spawn",
    )

    # raw mode
    p.add_argument("--max_samples", type=int, default=None)

    # trainer-like mode
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=5000)
    p.add_argument("--full_epoch", action="store_true", help="读取完整 1 个 epoch（忽略 --max_batches）")
    p.add_argument("--epochs", type=int, default=1, help="读取多少个 epoch（仅对 --full_epoch 或 max_batches 很大时有意义）")
    p.add_argument("--drop_last", action="store_true", help="模拟训练时 drop_last=True 的行为")
    p.add_argument("--persistent_workers", action="store_true", help="开启 persistent_workers（num_workers>0 时有效）")
    p.add_argument("--strict_expected", action="store_true", help="在 full_epoch 且 epochs=1 且不 drop_last 时，强制校验样本数=LeRobotValueDataset.__len__()")
    p.add_argument("--model_max_length", type=int, default=4096)

    # processor pixels (对齐训练默认值/可覆盖)
    p.add_argument("--min_pixels", type=int, default=28 * 28 * 16)
    p.add_argument("--max_pixels", type=int, default=28 * 28 * 576)
    p.add_argument("--video_min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--video_max_pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--video_min_frames", type=int, default=4)
    p.add_argument("--video_max_frames", type=int, default=8)
    p.add_argument("--video_fps", type=float, default=2.0)

    args = p.parse_args()

    if args.mp_start_method is not None:
        try:
            torch.multiprocessing.set_start_method(args.mp_start_method, force=True)
            print(f"[main] torch multiprocessing start_method={args.mp_start_method}")
        except RuntimeError as e:
            # start_method 只能设一次；如果上层已经设置过，这里打印提示即可
            print(f"[main] WARN: set_start_method({args.mp_start_method}) failed: {e}")

    # 一些 sanity check
    if args.num_workers == 0 and args.prefetch_factor is not None:
        # torch 会要求 num_workers>0 才能设置 prefetch_factor
        pass
    if args.buffer_size < 1:
        raise ValueError("--buffer_size must be >= 1 (0 会导致 shuffle buffer 逻辑报错)")

    if args.mode == "raw":
        return run_raw(args)
    return run_trainer_like(args)


if __name__ == "__main__":
    raise SystemExit(main())

