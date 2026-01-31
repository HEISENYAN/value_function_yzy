import json
import random
import logging
import re
import time
import itertools
import traceback
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

import transformers

from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels

    # --- Video Processor ---
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
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools - support both file paths (str) and PIL Image objects
    image_pool = []
    for img in images:
        try:
            image_pool.append({"type": "image", "image": img})
        except:
            raise ValueError(f"Unsupported image type: {type(img)}. Expected str (file path) or PIL.Image.Image")
    
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
    value_tokenizer=None,
) -> Dict:
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

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)

    # Only process value tokens if value_tokenizer is provided
    if value_tokenizer is not None:
        for pos in range(L):
            token_id = input_ids_flat[pos]

            # Check if this token is a value token (<extra_id_0> to <extra_id_200>)
            if token_id in value_tokenizer.extra_id_token_ids:
                labels[0, pos] = input_ids[0, pos]

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
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
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if concat_images is not None:
            batch["pixel_values"] = concat_images
            batch["image_grid_thw"] = grid_thw

        if concat_videos is not None:
            batch["pixel_values_videos"] = concat_videos
            batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
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
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if concat_images is not None:
            batch["pixel_values"] = concat_images
            batch["image_grid_thw"] = grid_thw

        if concat_videos is not None:
            batch["pixel_values_videos"] = concat_videos
            batch["video_grid_thw"] = video_grid_thw

        return batch


class MultiDatasetIterator:
    """
    Efficient iterator for multiple IterableSupervisedDataset instances.
    Uses round-robin sampling to interleave data from different datasets.
    """

    def __init__(self, datasets_with_weights):
        """
        Args:
            datasets_with_weights: List of (dataset, weight) or (dataset, weight, size) tuples
        """
        self.datasets = [ds for ds, _, _ in datasets_with_weights]
        self.weights = [weight for _, weight, _ in datasets_with_weights]
        self.sizes = [size for _, _, size in datasets_with_weights]
        self.iterators = []

        print(f"Multi-dataset iterator initialized with {len(self.datasets)} datasets:")
        for i, (ds, weight, size) in enumerate(datasets_with_weights):
            size_info = f"~{size} samples" if size is not None else "unknown size"
            print(f"  Dataset {i+1}: {size_info} (weight: {weight:.3f})")

    def __iter__(self):
        # Create fresh iterators for each dataset
        self.iterators = [iter(ds) for ds in self.datasets]
        self.dataset_idx = 0
        return self

    def __next__(self):
        if not self.iterators:
            raise StopIteration

        # Try up to len(datasets) times to find an active iterator
        for _ in range(len(self.iterators)):
            current_idx = self.dataset_idx % len(self.iterators)

            try:
                # Try to get next item from current dataset
                item = next(self.iterators[current_idx])

                # Move to next dataset for round-robin
                self.dataset_idx += 1
                return item

            except StopIteration:
                # This dataset is exhausted, remove it
                print(f"Dataset {current_idx + 1} exhausted, removing from rotation")
                self.iterators.pop(current_idx)
                self.weights.pop(current_idx)

                # Adjust current index if necessary
                if self.iterators and current_idx <= self.dataset_idx % (len(self.iterators) + 1):
                    self.dataset_idx -= 1

        # All datasets exhausted
        raise StopIteration


class MultiSupervisedDataset(IterableDataset):
    """
    Wrapper for multiple IterableSupervisedDataset instances with efficient round-robin sampling.
    """

    def __init__(self, datasets_with_weights):
        super().__init__()
        self.multi_iterator = MultiDatasetIterator(datasets_with_weights)

    def __iter__(self):
        return iter(self.multi_iterator)

    def __len__(self):
        # Approximate total length using stored sizes
        total = sum(size for size in self.multi_iterator.sizes if size is not None)
        return total if total > 0 else 0


class IterableSupervisedDataset(IterableDataset):
    """
    Qwen Adapter for Iterable Datasets.
    Wraps the LeRobotValueDataset (which yields PIL images) and applies Qwen preprocessing.
    """

    def __init__(self, iterable_dataset, processor, data_args, value_tokenizer=None):
        super(IterableSupervisedDataset, self).__init__()
        self.iterable_dataset = iterable_dataset
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.value_tokenizer = value_tokenizer

        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

        # Determine item processing function
        if getattr(data_args, "data_packing", False):
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item
    
    def __iter__(self):
        """Iterate over the wrapped dataset and apply Qwen preprocessing."""
        iter_dataset = iter(self.iterable_dataset)
        for item in iter_dataset:
            try:
                # Wrap item in list format expected by preprocess_qwen_visual
                sources = [item]
                processed_item = self.item_fn(sources)
                yield processed_item
            except Exception as e:
                # Catch image corruption or processing errors
                logging.warning(f"Error processing item in IterableSupervisedDataset: {e}")
                traceback.print_exc()
                continue
    
    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        """Process a single item through Qwen preprocessing pipeline."""
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            value_tokenizer=self.value_tokenizer,
        )
        
        seq_len = data_dict["input_ids"][0].size(0)
        
        # Handle image grid dimensions
        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]


            # Validate image_grid_thw: ensure sequence length is sufficient
            # spatial_merge_unit is typically 4, so we need at least 4 tokens
            if grid_thw:
                spatial_merge_unit = getattr(self.processor.image_processor, "merge_size", 2) ** 2
                for g in grid_thw:
                    if isinstance(g, torch.Tensor):
                        # Check if the total sequence length (prod of grid_thw) is sufficient
                        total_tokens = g.prod().item() if g.numel() > 0 else 0
                        if total_tokens > 0 and total_tokens < spatial_merge_unit:
                            # Skip this sample or raise error
                            raise ValueError(
                                f"Image sequence length {total_tokens} is too small "
                                f"(minimum required: {spatial_merge_unit}). "
                                f"grid_thw: {g.tolist()}"
                            )
        else:
            grid_thw = None
        
        # Handle video grid dimensions
        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None
        
        # Compute position IDs using RoPE (Rotary Position Embedding)
        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        # Text decoding and label processing
        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict
    
    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:
        """Process packed items (when data_packing=True)."""
        if isinstance(sources, dict):
            sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def make_supervised_data_module(processor, data_args, model_args=None, value_tokenizer=None) -> Dict:
    """
    Construct the dataset and collator, integrating the new LeRobotValueDataset.
    """
    from qwenvl.utils.value_tokenizer import ValueTokenizer
    from .data_loader_yzy import LeRobotValueDataset

    # Distributed Setup
    global local_rank
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Initialize Value Tokenizer
    if data_args.use_value_tokenizer and value_tokenizer is None:
        if model_args is None:
            raise ValueError("model_args is required when use_value_tokenizer=True")
        value_tokenizer = ValueTokenizer(
            llm_path=model_args.model_name_or_path,
            bins=data_args.value_tokenizer_bins,
            min_value=data_args.value_tokenizer_min,
            max_value=data_args.value_tokenizer_max,
        )
        rank0_print(f"Init ValueTokenizer: bins={data_args.value_tokenizer_bins}")

    # Dataset Directory - 支持多个数据集
    dataset_paths = [path.strip() for path in data_args.dataset_use.split(',') if path.strip()]

    if len(dataset_paths) == 1:
        dataset_dir = dataset_paths[0]
        if not os.path.exists(dataset_dir):
            dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", dataset_dir))
            if not os.path.exists(dataset_dir):
                raise FileNotFoundError(f"Dataset path not found: {dataset_paths[0]}")

        rank0_print(f"Loading LeRobot dataset from: {dataset_dir}")
        le_train_dataset = LeRobotValueDataset(
            dataset_dir=dataset_dir,
            split="train",
            val_ratio=getattr(data_args, "val_ratio", 0.1),
            seed=getattr(data_args, "seed", 42) + local_rank,
            buffer_size=getattr(data_args, "buffer_size", 5000),
            camera_names=getattr(data_args, "camera_names", ["cam_high", "cam_left_wrist", "cam_right_wrist"]),
            value_tokenizer=value_tokenizer,
            max_episodes=getattr(data_args, "max_train_episodes", None)
        )

        # Wrap with Qwen preprocessor
        train_dataset = IterableSupervisedDataset(le_train_dataset, processor, data_args, value_tokenizer)
        
        # Create validation dataset (10% of data) for training process evaluation
        rank0_print(f"Creating validation dataset from training data")
        # le_val_dataset = LeRobotValueDataset(
        #     dataset_dir=dataset_dir,
        #     split="val",
        #     val_ratio=getattr(data_args, "val_ratio", 0.1),
        #     seed=getattr(data_args, "seed", 42) + local_rank,  # Same seed for consistent split
        #     buffer_size=getattr(data_args, "buffer_size", 5000),
        #     camera_names=getattr(data_args, "camera_names", ["cam_high", "cam_left_wrist", "cam_right_wrist"]),
        #     value_tokenizer=value_tokenizer,
        # )
        # val_dataset = IterableSupervisedDataset(le_val_dataset, processor, data_args, value_tokenizer)

    else:
        rank0_print(f"Loading {len(dataset_paths)} LeRobot datasets for joint training")

        datasets_with_weights = []
        val_datasets_with_weights = []
        total_samples = 0
        val_total_samples = 0

        for i, path in enumerate(dataset_paths):
            # Resolve path
            if not os.path.exists(path):
                resolved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", path))
                if not os.path.exists(resolved_path):
                    raise FileNotFoundError(f"Dataset path not found: {path}")
                dataset_dir = resolved_path
            else:
                dataset_dir = path

            rank0_print(f"  Dataset {i+1}: {dataset_dir}")

            # Create training dataset with different seed for diversity
            le_dataset = LeRobotValueDataset(
                dataset_dir=dataset_dir,
                split="train",
                val_ratio=getattr(data_args, "val_ratio", 0.1),
                seed=getattr(data_args, "seed", 42) + local_rank + i * 1000,  # Different seed per dataset
                buffer_size=getattr(data_args, "buffer_size", 5000),
                camera_names=getattr(data_args, "camera_names", ["cam_high", "cam_left_wrist", "cam_right_wrist"]),
                value_tokenizer=value_tokenizer,
                max_episodes=getattr(data_args, "max_train_episodes", None)
            )

            # Wrap with Qwen preprocessor
            supervised_dataset = IterableSupervisedDataset(le_dataset, processor, data_args, value_tokenizer)

            # Calculate weight based on dataset size
            dataset_size = len(le_dataset)
            total_samples += dataset_size
            datasets_with_weights.append((supervised_dataset, dataset_size, dataset_size))
            
            # Create validation dataset for this training dataset
            # le_val_dataset = LeRobotValueDataset(
            #     dataset_dir=dataset_dir,
            #     split="val",
            #     val_ratio=getattr(data_args, "val_ratio", 0.1),
            #     seed=getattr(data_args, "seed", 42) + local_rank + i * 1000,  # Same seed for consistent split
            #     buffer_size=getattr(data_args, "buffer_size", 5000),
            #     camera_names=getattr(data_args, "camera_names", ["cam_high", "cam_left_wrist", "cam_right_wrist"]),
            #     value_tokenizer=value_tokenizer,
            #     )

            # val_supervised_dataset = IterableSupervisedDataset(le_val_dataset, processor, data_args, value_tokenizer)
            
            # Calculate weight based on dataset size
            # val_dataset_size = len(le_val_dataset)
            # val_total_samples += val_dataset_size
            # val_datasets_with_weights.append((val_supervised_dataset, val_dataset_size, val_dataset_size))

        # Normalize weights (format: (dataset, weight, size))
        datasets_with_weights = [(ds, size/total_samples, size) for ds, _, size in datasets_with_weights]

        # Create multi-dataset wrapper
        train_dataset = MultiSupervisedDataset(datasets_with_weights)

        # Create multi-dataset wrapper for validation
        # val_datasets_with_weights = [(ds, size/val_total_samples, size) for ds, _, size in val_datasets_with_weights]
        # val_dataset = MultiSupervisedDataset(val_datasets_with_weights)

    # === Select Collator ===
    if getattr(data_args, "data_flatten", False) or getattr(data_args, "data_packing", False):
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)

    return dict(
        train_dataset=train_dataset,
        #eval_dataset=val_dataset,  # Use validation split from training data for training process evaluation
        data_collator=data_collator
    )
