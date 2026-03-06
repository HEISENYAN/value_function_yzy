import itertools
import logging
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import transformers
from collections.abc import Sequence as SequenceABC
from torch.utils.data import IterableDataset

from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3
from .data_loader_pair import LeRobotPairDataset

IGNORE_INDEX = -100

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    ip = processor.image_processor
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels

    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]
    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    image_pool = [{"type": "image", "image": img} for img in images]
    video_pool = [{"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]
        if role == "user":
            content = []
            text_parts = re.split(r"(<image>|<video>)", text)
            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError("Number of <image> placeholders exceeds provided images")
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError("Number of <video> placeholders exceeds provided videos")
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)")
    return messages


def preprocess_qwen_visual_pair(sources, processor) -> Dict[str, Any]:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    full_result["input_ids"] = input_ids
    full_result["labels"] = torch.full_like(input_ids, IGNORE_INDEX)
    full_result["delta_label"] = float(source["delta_label"])
    full_result["t_group_weight"] = float(source.get("t_group_weight", 1.0))
    return full_result


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensors.append(torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1))
    return torch.cat(padded_tensors, dim=1)


@dataclass
class PairDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        labels = [instance["labels"].squeeze(0) for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids,
            delta_labels=torch.tensor(
                [instance["delta_label"] for instance in instances], dtype=torch.float32
            ),
            t_group_weights=torch.tensor(
                [instance.get("t_group_weight", 1.0) for instance in instances], dtype=torch.float32
            ),
        )

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            batch["image_grid_thw"] = torch.cat(grid_thw, dim=0)

        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]
        if videos:
            batch["pixel_values_videos"] = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance
            ]
            batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        return batch


@dataclass
class FlattenedPairDataCollator(PairDataCollator):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(*([instance["attention_mask"] for instance in instances if "attention_mask" in instance]))
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
            delta_labels=torch.tensor(
                [instance["delta_label"] for instance in instances], dtype=torch.float32
            ),
            t_group_weights=torch.tensor(
                [instance.get("t_group_weight", 1.0) for instance in instances], dtype=torch.float32
            ),
        )

        if any("pixel_values" in d for d in instances):
            batch["pixel_values"] = torch.cat([d["pixel_values"] for d in instances if "pixel_values" in d], dim=0)
            batch["image_grid_thw"] = torch.cat([d["image_grid_thw"] for d in instances if "image_grid_thw" in d], dim=0)
        if any("pixel_values_videos" in d for d in instances):
            batch["pixel_values_videos"] = torch.cat(
                [d["pixel_values_videos"] for d in instances if "pixel_values_videos" in d], dim=0
            )
            batch["video_grid_thw"] = torch.cat([d["video_grid_thw"] for d in instances if "video_grid_thw" in d], dim=0)
        return batch


class IterableSupervisedPairDataset(IterableDataset):
    def __init__(self, iterable_dataset, processor, data_args, max_samples_per_epoch=None):
        super().__init__()
        self.iterable_dataset = iterable_dataset
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.max_samples_per_epoch = max_samples_per_epoch
        self.model_type = data_args.model_type
        if self.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif self.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif self.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.item_fn = self._get_packed_item if getattr(data_args, "data_packing", False) else self._get_item

    def __iter__(self):
        yielded = 0
        for item in iter(self.iterable_dataset):
            if self.max_samples_per_epoch is not None and yielded >= self.max_samples_per_epoch:
                break
            try:
                yield self.item_fn([item])
                yielded += 1
            except Exception as e:
                logging.warning(f"Error processing pair item: {e}")
                traceback.print_exc()
                continue

    def __len__(self):
        if hasattr(self.iterable_dataset, "__len__"):
            base_len = len(self.iterable_dataset)
            if self.max_samples_per_epoch is None:
                return base_len
            return min(base_len, self.max_samples_per_epoch)
        raise TypeError("Underlying iterable_dataset does not implement __len__")

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual_pair(sources, self.processor)
        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, SequenceABC):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, SequenceABC):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=torch.cat(video_grid_thw, dim=0) if video_grid_thw else None,
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:
        if isinstance(sources, dict):
            sources = [sources]
        assert len(sources) == 1, "Packed pair mode expects wrapped single source entries."

        data_list = []
        for source in sources:
            source = [source] if isinstance(source, dict) else source
            data_list.append(self._get_item(source))

        new_data_dict = {
            "input_ids": torch.cat([d["input_ids"] for d in data_list], dim=1),
            "labels": torch.cat([d["labels"] for d in data_list], dim=1),
            "position_ids": torch.cat([d["position_ids"] for d in data_list], dim=2),
            "attention_mask": [d["attention_mask"][0] for d in data_list if "attention_mask" in d],
            "delta_label": data_list[0]["delta_label"],
            "t_group_weight": data_list[0]["t_group_weight"],
        }

        if any("pixel_values" in d for d in data_list):
            new_data_dict["pixel_values"] = torch.cat(
                [d["pixel_values"] for d in data_list if "pixel_values" in d], dim=0
            )
            new_data_dict["image_grid_thw"] = torch.cat(
                [d["image_grid_thw"] for d in data_list if "image_grid_thw" in d], dim=0
            )
        if any("pixel_values_videos" in d for d in data_list):
            new_data_dict["pixel_values_videos"] = torch.cat(
                [d["pixel_values_videos"] for d in data_list if "pixel_values_videos" in d], dim=0
            )
            new_data_dict["video_grid_thw"] = torch.cat(
                [d["video_grid_thw"] for d in data_list if "video_grid_thw" in d], dim=0
            )
        return new_data_dict


class MultiDatasetIterator:
    def __init__(self, datasets_with_sizes):
        self.datasets = [ds for ds, _ in datasets_with_sizes]
        self.sizes = [size for _, size in datasets_with_sizes]
        self.iterators = []

    def __iter__(self):
        self.iterators = [iter(ds) for ds in self.datasets]
        self.dataset_idx = 0
        return self

    def __next__(self):
        if not self.iterators:
            raise StopIteration
        for _ in range(len(self.iterators)):
            current_idx = self.dataset_idx % len(self.iterators)
            try:
                item = next(self.iterators[current_idx])
                self.dataset_idx += 1
                return item
            except StopIteration:
                self.iterators.pop(current_idx)
                if self.iterators and current_idx <= self.dataset_idx % (len(self.iterators) + 1):
                    self.dataset_idx -= 1
        raise StopIteration


class MultiSupervisedPairDataset(IterableDataset):
    def __init__(self, datasets_with_sizes):
        super().__init__()
        self.multi_iterator = MultiDatasetIterator(datasets_with_sizes)

    def __iter__(self):
        return iter(self.multi_iterator)

    def __len__(self):
        return sum(self.multi_iterator.sizes)


def _resolve_dataset_path(path: str) -> str:
    if os.path.exists(path):
        return path
    resolved = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", path))
    if os.path.exists(resolved):
        return resolved
    raise FileNotFoundError(f"Dataset path not found: {path}")


def _sync_min_len_across_ranks(local_len: int) -> int:
    if not torch.distributed.is_initialized():
        return int(local_len)

    backend = torch.distributed.get_backend()
    if backend == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    t = torch.tensor([int(local_len)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
    return int(t.item())


def make_supervised_data_module(processor, data_args, model_args=None, value_tokenizer=None) -> Dict:
    global local_rank
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    data_args.model_type = getattr(data_args, "model_type", "qwen2.5vl")

    dataset_paths = [path.strip() for path in data_args.dataset_use.split(",") if path.strip()]
    if not dataset_paths:
        raise ValueError("dataset_use is empty.")

    def create_pair_dataset(dataset_dir: str, split: str, seed_offset: int = 0):
        return LeRobotPairDataset(
            dataset_dir=dataset_dir,
            split=split,
            val_ratio=getattr(data_args, "val_ratio", 0.1),
            seed=getattr(data_args, "seed", 42) + local_rank + seed_offset,
            buffer_size=getattr(data_args, "buffer_size", 5000),
            camera_names=getattr(data_args, "camera_names", ["cam_high", "cam_left_wrist", "cam_right_wrist"]),
            max_episodes=getattr(data_args, "max_train_episodes", None) if split == "train" else None,
            pair_short_step=getattr(data_args, "pair_short_step", 8),
            pair_mid_step=getattr(data_args, "pair_mid_step", 16),
            pair_random_min=getattr(data_args, "pair_random_min", 1),
            pair_add_backward=getattr(data_args, "pair_add_backward", True),
            pair_prompt_style=getattr(data_args, "pair_prompt_style", "explicit_t0_t1"),
            rank=local_rank,
            world_size=world_size,
        )

    if len(dataset_paths) == 1:
        dataset_dir = _resolve_dataset_path(dataset_paths[0])
        rank0_print(f"Loading pair dataset from: {dataset_dir}")
        le_train = create_pair_dataset(dataset_dir, "train")
        le_val = create_pair_dataset(dataset_dir, "val")

        train_min_len = _sync_min_len_across_ranks(len(le_train))
        eval_min_len = _sync_min_len_across_ranks(len(le_val))
        rank0_print(
            f"[PairDataset] synchronized per-rank lengths: train_min_len={train_min_len}, eval_min_len={eval_min_len}"
        )

        train_dataset = IterableSupervisedPairDataset(
            le_train, processor, data_args, max_samples_per_epoch=train_min_len
        )
        eval_dataset = IterableSupervisedPairDataset(
            le_val, processor, data_args, max_samples_per_epoch=eval_min_len
        )
    else:
        train_parts = []
        val_parts = []
        for i, path in enumerate(dataset_paths):
            dataset_dir = _resolve_dataset_path(path)
            rank0_print(f"Loading pair dataset {i+1}: {dataset_dir}")
            le_train = create_pair_dataset(dataset_dir, "train", seed_offset=i * 1000)
            le_val = create_pair_dataset(dataset_dir, "val", seed_offset=i * 1000)
            train_parts.append((IterableSupervisedPairDataset(le_train, processor, data_args), len(le_train)))
            val_parts.append((IterableSupervisedPairDataset(le_val, processor, data_args), len(le_val)))
        train_dataset = MultiSupervisedPairDataset(train_parts)
        eval_dataset = MultiSupervisedPairDataset(val_parts)

    if getattr(data_args, "data_flatten", False) or getattr(data_args, "data_packing", False):
        data_collator = FlattenedPairDataCollator(processor.tokenizer)
    else:
        data_collator = PairDataCollator(processor.tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
