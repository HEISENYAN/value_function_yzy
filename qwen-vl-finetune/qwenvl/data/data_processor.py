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
from torch.utils.data import Dataset, IterableDataset

import transformers

from . import data_list, detect_dataset_type, create_value_dataset
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3
from .datasets import OpenXValueDataset, RoboTwinValueDataset, OpenPiValueDataset

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # If image_size is specified, use fixed size instead of pixel range
    if data_args.image_size is not None:
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = data_args.image_size
            ip.size["longest_edge"] = data_args.image_size
            rank0_print(f"✅ Updated image_processor to fixed size {data_args.image_size}x{data_args.image_size}")
        # Disable dynamic sizing
        if hasattr(ip, "min_pixels"):
            ip.min_pixels = data_args.image_size * data_args.image_size
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = data_args.image_size * data_args.image_size

    else:
        # Use dynamic pixel range
        if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
            ip.min_pixels = data_args.min_pixels
            ip.max_pixels = data_args.max_pixels
            rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
            rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = data_args.min_pixels
            ip.size["longest_edge"] = data_args.max_pixels
            rank0_print(
                f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
            )
            rank0_print(
                f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
            )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

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
    from PIL import Image as PILImage
    image_pool = []
    for img in images:
        if isinstance(img, PILImage.Image):
            # Direct PIL Image object - pass it directly
            image_pool.append({"type": "image", "image": img})
        elif isinstance(img, str):
            # File path - convert to absolute path
            image_pool.append({"type": "image", "image": _make_abs_paths(base_path, img)})
        else:
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

    # Directly find value tokens in the sequence
    label_positions_found = []

    # Only process value tokens if value_tokenizer is provided
    if value_tokenizer is not None:
        for pos in range(L):
            token_id = input_ids_flat[pos]

            # Check if this token is a value token (<extra_id_0> to <extra_id_200>)
            if token_id in value_tokenizer.extra_id_token_ids:
                labels[0, pos] = input_ids[0, pos]
                label_positions_found.append(pos)

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args, value_tokenizer=None):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.value_tokenizer = value_tokenizer
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            value_tokenizer=self.value_tokenizer,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

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

        if isinstance(sources, dict):
            sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
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


class QwenIterableDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset wrapper that applies Qwen preprocessing to items from IterableDataset.
    
    This class wraps an IterableDataset (e.g., OpenXValueDataset or RoboTwinValueDataset)
    and applies Qwen's preprocessing pipeline (preprocess_qwen_visual) to each item.
    
    The wrapped IterableDataset should yield items in Qwen-compatible format:
    {
        "conversations": [...],
        "image": [PIL.Image, ...] or ["path1", "path2", ...],
        "data_path": "..."
    }
    """
    
    def __init__(self, iterable_dataset, processor, data_args, value_tokenizer=None):
        """
        Initialize the Qwen IterableDataset wrapper.

        Args:
            iterable_dataset: An IterableDataset that yields Qwen-compatible format items
            processor: Qwen processor instance
            data_args: Data arguments containing model_type and other settings
            value_tokenizer: ValueTokenizer instance for encoding R values
        """
        super(QwenIterableDataset, self).__init__()
        self.iterable_dataset = iterable_dataset
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.value_tokenizer = value_tokenizer
        
        # Set up model_type and get_rope_index (for position encoding)
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

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item
    
    def __iter__(self):
        """Iterate over the wrapped dataset and apply Qwen preprocessing."""
        for item in self.iterable_dataset:
            try:
                # Wrap item in list format expected by preprocess_qwen_visual
                sources = [item]
                processed_item = self.item_fn(sources)
                yield processed_item
            except Exception as e:
                logging.warning(f"Error processing item in QwenIterableDataset: {e}")
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

        # Text decoding and label processing (for consistency with LazySupervisedDataset)
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


def _create_value_dataset_from_path(data_args, model_args, processor):
    """
    Create a value dataset from dataset path using unified factory.

    Args:
        data_args: Data arguments
        model_args: Model arguments (for value_tokenizer)
        processor: Qwen processor

    Returns:
        IterableDataset instance or None if not applicable
    """
    import torch
    from qwenvl.utils.value_tokenizer import ValueTokenizer
    from . import get_dataset_config

    # Get distributed training info
    global local_rank
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Get dataset configuration
    dataset_config = get_dataset_config(data_args.dataset_use)

    # Skip RLDS-based datasets (handled separately)
    if dataset_config.get("requires_rlds", False):
        return None

    # Create value_tokenizer if enabled
    value_tokenizer = None
    if data_args.use_value_tokenizer:
        if model_args is None:
            raise ValueError("model_args is required when use_value_tokenizer=True")
        value_tokenizer = ValueTokenizer(
            llm_path=model_args.model_name_or_path,
            bins=data_args.value_tokenizer_bins,
            min_value=data_args.value_tokenizer_min,
            max_value=data_args.value_tokenizer_max,
        )
        rank0_print(f"Created ValueTokenizer with bins={data_args.value_tokenizer_bins}, "
                   f"range=[{data_args.value_tokenizer_min}, {data_args.value_tokenizer_max}]")

    # Resolve dataset directory
    dataset_dir = data_args.dataset_use
    if not os.path.isabs(dataset_dir):
        abs_path = os.path.abspath(dataset_dir)
        if os.path.exists(abs_path):
            dataset_dir = abs_path
        else:
            # Fallback: assume relative to project root (qwen-vl-finetune)
            project_root = Path(__file__).parent.parent.parent.parent
            dataset_dir = str(project_root / dataset_dir)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rank0_print(f"Creating {dataset_config['dataset_type']} dataset from: {dataset_dir}")

    # Get num_workers from training_args if available, otherwise use default
    num_workers = getattr(data_args, 'dataloader_num_workers', 4)

    # Create dataset using factory function
    iterable_dataset = create_value_dataset(
        dataset_config,
        dataset_name=dataset_config["dataset_type"].lower(),
        transform=None,  # Not used for value datasets
        tokenizer=processor.tokenizer,
        dataset_dir=dataset_dir,
        local_rank=local_rank,
        world_size=world_size,
        num_workers=num_workers,
        value_tokenizer=value_tokenizer,
    )

    return iterable_dataset


def _create_oxe_dataset(data_args, model_args, processor):
    """
    Create OpenX Embodiment dataset (special handling due to RLDS dependency).

    Args:
        data_args: Data arguments
        model_args: Model arguments
        processor: Qwen processor

    Returns:
        OpenXValueDataset instance or None
    """
    import torch
    from qwenvl.utils.value_tokenizer import ValueTokenizer

    # Get distributed training info
    global local_rank
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Create value_tokenizer if enabled
    value_tokenizer = None
    if data_args.use_value_tokenizer:
        if model_args is None:
            raise ValueError("model_args is required when use_value_tokenizer=True")
        value_tokenizer = ValueTokenizer(
            llm_path=model_args.model_name_or_path,
            bins=data_args.value_tokenizer_bins,
            min_value=data_args.value_tokenizer_min,
            max_value=data_args.value_tokenizer_max,
        )
        rank0_print(f"Created ValueTokenizer with bins={data_args.value_tokenizer_bins}, "
                   f"range=[{data_args.value_tokenizer_min}, {data_args.value_tokenizer_max}]")

    # For OpenX, we need to parse the dataset_use to extract data_mix and data_root_dir
    # This is a simplified version - in practice, you might need more sophisticated parsing
    dataset_use = data_args.dataset_use
    if ":" in dataset_use:
        # Format: "data_root_dir:data_mix"
        data_root_dir, data_mix = dataset_use.split(":", 1)
    else:
        # Default values
        data_root_dir = "/path/to/openx/data"
        data_mix = "bridge_rt_1"

    rank0_print(f"Creating OpenXValueDataset with data_mix={data_mix}")

    iterable_dataset = OpenXValueDataset(
        dataset_name="open_x_embodiment",
        transform=None,
        tokenizer=processor.tokenizer,
        data_dir_list=[],  # Will be filled by RLDS
        data_root_dir=Path(data_root_dir),
        data_mix=data_mix,
        resize_resolution=(256, 256),
        local_rank=local_rank,
        world_size=world_size,
        num_workers=getattr(data_args, 'dataloader_num_workers', 8),
        data_status=None,
        shuffle_buffer_size=256_000,
        train=True,
        image_aug=False,
        value_tokenizer=value_tokenizer,
    )

    return iterable_dataset


def make_supervised_data_module(processor, data_args, model_args=None, value_tokenizer=None) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    
    This function supports both the legacy map-style Dataset (LazySupervisedDataset) 
    and the new IterableDataset format (QwenIterableDataset).
    
    Args:
        processor: Qwen processor instance
        data_args: Data arguments containing model_type and other settings
        iterable_dataset: Optional IterableDataset (e.g., OpenXActionFlowDataset or 
                        RoboTwinActionFlowDataset) that yields items in Qwen-compatible format:
                        {
                            "conversations": [
                                {"from": "human", "value": You are estimating task progress for robotic manipulation.<image>},
                                {"from": "gpt", "value": "[action as text]"}
                            ],
                            "image": [PIL.Image, ...] or ["path1", "path2", ...],
                            "data_path": "..."  # Optional, only needed if using file paths
                        }
                        If provided, will use QwenIterableDataset wrapper.
                        If None, will use legacy LazySupervisedDataset (backward compatible).
        model_args: Optional ModelArguments containing model_name_or_path for value_tokenizer
    
    Returns:
        Dictionary with train_dataset, eval_dataset (None), and data_collator
    """
    dataset_type = detect_dataset_type(data_args.dataset_use)

    if dataset_type == "oxe":
        iterable_dataset = _create_oxe_dataset(data_args, model_args, processor)
    elif dataset_type in ["robottwin", "openpi"]:
        iterable_dataset = _create_value_dataset_from_path(data_args, model_args, processor)

    if iterable_dataset is not None:
        if not isinstance(iterable_dataset, IterableDataset):
            raise TypeError(
                f"iterable_dataset must be an IterableDataset, got {type(iterable_dataset)}. "
                "Please use an IterableDataset such as OpenXActionFlowDataset or RoboTwinActionFlowDataset."
            )
        train_dataset = QwenIterableDataset(iterable_dataset, processor, data_args, value_tokenizer=value_tokenizer)
    else:
        # Legacy format: Use map-style LazySupervisedDataset (backward compatible)
        train_dataset = LazySupervisedDataset(processor, data_args=data_args, value_tokenizer=value_tokenizer)

    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
