from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None
    Qwen3VLMoeForConditionalGeneration = None

from qwenvl.data.data_loader_pi06_map import VALUE_PROMPT_TEMPLATE
from qwenvl.data.data_processor_pair_map import PairBatchBuilder, update_processor_pixels
from qwenvl.train.pair_model import load_pair_model
from qwenvl.utils.value_tokenizer import ValueTokenizer

logger = logging.getLogger(__name__)

CAMERA_ALIASES = {
    "cam_high": ["cam_high", "top_head"],
    "cam_left_wrist": ["cam_left_wrist", "hand_left"],
    "cam_right_wrist": ["cam_right_wrist", "hand_right"],
    "top_head": ["top_head", "cam_high"],
    "hand_left": ["hand_left", "cam_left_wrist"],
    "hand_right": ["hand_right", "cam_right_wrist"],
}

_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")


def load_run_config(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    for candidate in (model_dir / "run_config.json", model_dir.parent / "run_config.json"):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as fin:
                return json.load(fin)
    return {}


def resolve_checkpoint_path(checkpoint_dir: str | Path, checkpoint_step: str) -> Path:
    checkpoint_root = Path(checkpoint_dir)
    step = str(checkpoint_step)

    def has_weights(path: Path) -> bool:
        return any(
            (path / name).exists()
            for name in (
                "pair_model.bin",
                "pytorch_model.bin",
                "model.safetensors",
                "adapter_model.bin",
                "adapter_model.safetensors",
            )
        )

    candidates: list[Path] = []
    if step in {"final", "final_model"}:
        candidates.extend([checkpoint_root / "final_model", checkpoint_root])
    else:
        candidates.extend([checkpoint_root / f"checkpoint-{step}", checkpoint_root / step])
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and has_weights(candidate):
            return candidate
    if step in {"final", "final_model"} and checkpoint_root.exists() and has_weights(checkpoint_root):
        return checkpoint_root
    raise FileNotFoundError(
        f"Could not resolve checkpoint path from checkpoint_dir={checkpoint_root}, checkpoint_step={checkpoint_step}"
    )


def find_processor_dir(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    candidates = [
        model_dir,
        model_dir.parent,
        model_dir.parent / "final_model",
    ]
    marker_files = ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json")
    for candidate in candidates:
        if candidate.exists() and any((candidate / marker).exists() for marker in marker_files):
            return candidate
    return model_dir


def format_lerobot_path(
    pattern: str,
    episode_idx: int,
    chunks_size: int,
    video_key: Optional[str] = None,
) -> Path:
    chunk_index = int(episode_idx) // int(chunks_size)
    format_kwargs = {
        "episode_chunk": chunk_index,
        "episode_index": int(episode_idx),
        "chunk_index": chunk_index,
        "file_index": int(episode_idx),
    }
    if video_key is not None:
        format_kwargs["video_key"] = video_key
    return Path(pattern.format(**format_kwargs))


def resolve_video_paths(repo_root: Path, metadata, episode_idx: int, camera_names: list[str]) -> tuple[Path, Path, Path]:
    if len(camera_names) != 3:
        raise ValueError(f"Expected exactly 3 camera names, got {camera_names}")
    resolved_paths = []
    video_pattern = getattr(metadata, "video_path")
    chunks_size = int(getattr(metadata, "chunks_size"))
    for camera_name in camera_names:
        candidates = CAMERA_ALIASES.get(camera_name, [camera_name])
        matched_path = None
        tried = []
        for candidate_key in candidates:
            rel_path = format_lerobot_path(video_pattern, episode_idx, chunks_size, video_key=candidate_key)
            full_path = repo_root / rel_path
            tried.append(str(full_path))
            if full_path.exists():
                matched_path = full_path
                break
        if matched_path is None:
            raise FileNotFoundError(
                f"Missing video for episode={episode_idx}, camera={camera_name}. Tried: {tried}"
            )
        resolved_paths.append(matched_path)
    return tuple(resolved_paths)  # type: ignore[return-value]


def resolve_parquet_path(repo_root: Path, metadata, episode_idx: int) -> Path:
    rel_path = format_lerobot_path(
        getattr(metadata, "data_path"),
        episode_idx=episode_idx,
        chunks_size=int(getattr(metadata, "chunks_size")),
    )
    return repo_root / rel_path


def _normalize_instruction(raw_instruction: Any) -> Optional[str]:
    if isinstance(raw_instruction, list):
        raw_instruction = raw_instruction[0] if raw_instruction else None
    if raw_instruction is None:
        return None
    return str(raw_instruction)


def _episode_meta(metadata, episode_idx: int) -> dict[str, Any]:
    episodes = getattr(metadata, "episodes", None)
    if isinstance(episodes, dict):
        return episodes.get(str(episode_idx), episodes.get(episode_idx, {}))
    if isinstance(episodes, list):
        return episodes[episode_idx]
    return {}


def resolve_prompt(metadata, episode_idx: int, prompt_override: Optional[str]) -> str:
    if prompt_override:
        return prompt_override
    episode_meta = _episode_meta(metadata, episode_idx)
    prompt = _normalize_instruction(episode_meta.get("tasks")) or _normalize_instruction(episode_meta.get("task"))
    if prompt:
        return prompt
    tasks = getattr(metadata, "tasks", None)
    if tasks:
        task_index = episode_meta.get("task_index")
        if task_index is not None and 0 <= int(task_index) < len(tasks):
            return str(tasks[int(task_index)])
        if len(tasks) == 1:
            return str(tasks[0])
    raise ValueError(
        f"Could not resolve prompt for episode {episode_idx}. "
        "Pass --prompt explicitly or ensure metadata contains task/tasks."
    )


def load_lerobot_metadata(repo_root: str | Path):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    return LeRobotDatasetMetadata(str(repo_root))


def _load_qwen_generation_model(
    model_path: str | Path,
    attn_implementation: Optional[str] = None,
    bf16: bool = False,
):
    model_path = str(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model_type = str(getattr(config, "model_type", "")).lower()
    dtype = torch.bfloat16 if bf16 else None
    kwargs = {
        "attn_implementation": attn_implementation,
        "torch_dtype": dtype,
    }
    if model_type == "qwen3_vl_moe":
        if Qwen3VLMoeForConditionalGeneration is None:
            raise ImportError("Current transformers build does not provide Qwen3VLMoeForConditionalGeneration.")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif model_type == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError("Current transformers build does not provide Qwen3VLForConditionalGeneration.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif model_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported pi06 model_type={model_type!r} from checkpoint={model_path}")
    return model, model_type


class BaseRolloutEvaluator(ABC):
    evaluator_model_type = "base"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        batch_size: int = 8,
        num_workers: int = 8,
        device: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._executor = ThreadPoolExecutor(max_workers=max(1, self.num_workers))

    def shutdown(self):
        self._executor.shutdown(wait=True)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def _load_video_frames(self, video_path: str | Path, frame_interval: int = 1) -> list[np.ndarray]:
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frames: list[np.ndarray] = []
        frame_count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_count % max(1, int(frame_interval)) == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
        finally:
            cap.release()
        return frames

    def _load_videos_parallel(
        self,
        video_paths: tuple[str | Path, str | Path, str | Path],
        frame_interval: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        futures = {
            self._executor.submit(self._load_video_frames, path, frame_interval): idx
            for idx, path in enumerate(video_paths)
        }
        results: dict[int, list[np.ndarray]] = {}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return results[0], results[1], results[2]

    @staticmethod
    def _slice_frames(
        frames: list[np.ndarray],
        min_frame_index: Optional[int],
        max_frame_index: Optional[int],
    ) -> list[np.ndarray]:
        start = 0 if min_frame_index is None else int(min_frame_index)
        end = None if max_frame_index is None else int(max_frame_index) + 1
        return frames[start:end]

    @staticmethod
    def _frame_triplet_to_images(frame_triplet: tuple[np.ndarray, np.ndarray, np.ndarray]) -> list[Image.Image]:
        return [Image.fromarray(frame, mode="RGB") for frame in frame_triplet]

    @staticmethod
    def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.to(device, non_blocking=True)
            else:
                out[key] = value
        return out

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _build_metric_rows(
        self,
        absolute_values: np.ndarray,
        relative_interval: int,
        relative_values: Optional[np.ndarray] = None,
        episode_idx: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        relative_interval = max(1, int(relative_interval))
        num_frames = int(len(absolute_values))
        future_indices = np.minimum(np.arange(num_frames) + relative_interval, max(0, num_frames - 1))
        rows: list[dict[str, Any]] = []

        for frame_idx in range(num_frames):
            future_idx = int(future_indices[frame_idx])
            gap = future_idx - frame_idx
            if gap <= 0:
                derived_adv = 0.0
            else:
                derived_adv = float(absolute_values[future_idx] - absolute_values[frame_idx])
                if gap != relative_interval:
                    derived_adv = derived_adv / gap * relative_interval

            if relative_values is None:
                relative_adv = derived_adv
            else:
                if gap <= 0:
                    relative_adv = 0.0
                else:
                    relative_adv = float(relative_values[frame_idx])
                    if gap != relative_interval:
                        relative_adv = relative_adv / gap * relative_interval

            row = {
                "frame_idx": int(frame_idx),
                "future_frame_idx": int(future_idx),
                "absolute_value": self._clip_unit(float(absolute_values[frame_idx])),
                "relative_advantage": self._clip_unit(relative_adv),
                "absolute_advantage": self._clip_unit(derived_adv),
                "model_type": self.evaluator_model_type,
            }
            if episode_idx is not None:
                row["episode_idx"] = int(episode_idx)
            rows.append(row)
        return rows

    @abstractmethod
    def _predict_metric_arrays(
        self,
        top_frames: list[np.ndarray],
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        prompt: str,
        relative_interval: int,
        prefetch: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    @torch.inference_mode()
    def evaluate_video_metrics(
        self,
        video_paths: tuple[str | Path, str | Path, str | Path],
        prompt: str,
        relative_interval: int = 50,
        frame_interval: int = 1,
        min_frame_index: Optional[int] = None,
        max_frame_index: Optional[int] = None,
        prefetch: bool = True,
        episode_idx: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        top_frames, left_frames, right_frames = self._load_videos_parallel(video_paths, frame_interval)
        if not (len(top_frames) == len(left_frames) == len(right_frames)):
            raise ValueError(
                "Inconsistent frame counts across cameras: "
                f"top={len(top_frames)}, left={len(left_frames)}, right={len(right_frames)}"
            )

        top_frames = self._slice_frames(top_frames, min_frame_index, max_frame_index)
        left_frames = self._slice_frames(left_frames, min_frame_index, max_frame_index)
        right_frames = self._slice_frames(right_frames, min_frame_index, max_frame_index)
        if not top_frames:
            raise ValueError("No frames remain after slicing.")

        absolute_values, relative_values = self._predict_metric_arrays(
            top_frames=top_frames,
            left_frames=left_frames,
            right_frames=right_frames,
            prompt=prompt,
            relative_interval=relative_interval,
            prefetch=prefetch,
        )
        return self._build_metric_rows(
            absolute_values=absolute_values,
            relative_interval=relative_interval,
            relative_values=relative_values,
            episode_idx=episode_idx,
        )


class PairRolloutEvaluator(BaseRolloutEvaluator):
    evaluator_model_type = "pair"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name_or_path: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        attn_implementation: str = "flash_attention_2",
        bf16: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        run_config = load_run_config(self.checkpoint_dir)
        base_model_name = model_name_or_path
        if base_model_name is None:
            base_model_name = (
                run_config.get("model_args", {}).get("model_name_or_path")
                or run_config.get("base_model_name_or_path")
            )

        self.model, self.model_meta = load_pair_model(
            checkpoint_dir=self.checkpoint_dir,
            model_name_or_path=base_model_name,
            attn_implementation=attn_implementation,
            bf16=bf16 and self.device.type == "cuda",
            map_location="cpu",
        )
        self.model.to(self.device)
        self.model.eval()

        processor_dir = find_processor_dir(self.checkpoint_dir)
        self.processor = AutoProcessor.from_pretrained(processor_dir)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        data_args_snapshot = run_config.get("data_args", {})
        if data_args_snapshot:
            self.processor = update_processor_pixels(self.processor, SimpleNamespace(**data_args_snapshot))
        self.batch_builder = PairBatchBuilder(self.processor, self.model.model_type)

        logger.info(
            "PairRolloutEvaluator ready: device=%s checkpoint=%s processor=%s",
            self.device,
            self.checkpoint_dir,
            processor_dir,
        )

    def _prepare_pair_batches(
        self,
        top_frames: list[np.ndarray],
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        prompt: str,
        batch_start: int,
        relative_interval: int,
    ) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
        num_frames = len(top_frames)
        batch_end = min(batch_start + self.batch_size, num_frames)
        frame_indices = list(range(batch_start, batch_end))
        first_triplet = (top_frames[0], left_frames[0], right_frames[0])

        relative_examples = []
        absolute_examples = []
        for frame_idx in frame_indices:
            future_idx = min(frame_idx + relative_interval, num_frames - 1)
            current_triplet = (top_frames[frame_idx], left_frames[frame_idx], right_frames[frame_idx])
            future_triplet = (top_frames[future_idx], left_frames[future_idx], right_frames[future_idx])
            relative_examples.append(
                {
                    "instruction": prompt,
                    "current_images": self._frame_triplet_to_images(future_triplet),
                    "history_images": self._frame_triplet_to_images(current_triplet),
                }
            )
            absolute_examples.append(
                {
                    "instruction": prompt,
                    "current_images": self._frame_triplet_to_images(current_triplet),
                    "history_images": self._frame_triplet_to_images(first_triplet),
                }
            )
        relative_batch = self.batch_builder.build_batch(relative_examples)
        absolute_batch = self.batch_builder.build_batch(absolute_examples)
        return frame_indices, relative_batch, absolute_batch

    def _predict_metric_arrays(
        self,
        top_frames: list[np.ndarray],
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        prompt: str,
        relative_interval: int,
        prefetch: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        num_frames = len(top_frames)
        batch_starts = list(range(0, num_frames, self.batch_size))
        absolute_values = np.zeros(num_frames, dtype=np.float32)
        relative_values = np.zeros(num_frames, dtype=np.float32)

        def prepare(batch_start: int):
            return self._prepare_pair_batches(
                top_frames=top_frames,
                left_frames=left_frames,
                right_frames=right_frames,
                prompt=prompt,
                batch_start=batch_start,
                relative_interval=relative_interval,
            )

        prefetch_future = None
        if prefetch and batch_starts:
            prefetch_future = self._executor.submit(prepare, batch_starts[0])

        for batch_idx, batch_start in enumerate(batch_starts):
            if prefetch and prefetch_future is not None:
                frame_indices, relative_batch, absolute_batch = prefetch_future.result()
            else:
                frame_indices, relative_batch, absolute_batch = prepare(batch_start)

            if prefetch and batch_idx + 1 < len(batch_starts):
                prefetch_future = self._executor.submit(prepare, batch_starts[batch_idx + 1])

            relative_batch = self._move_batch_to_device(relative_batch, self.device)
            absolute_batch = self._move_batch_to_device(absolute_batch, self.device)
            relative_batch.pop("delta_labels", None)
            relative_batch.pop("t_group_weights", None)
            absolute_batch.pop("delta_labels", None)
            absolute_batch.pop("t_group_weights", None)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                relative_vals = self.model.sample_pair_delta(**relative_batch).detach().float().cpu().numpy()
                absolute_vals = self.model.sample_values(**absolute_batch).detach().float().cpu().numpy()

            absolute_values[np.asarray(frame_indices, dtype=np.int64)] = absolute_vals.astype(np.float32)
            relative_values[np.asarray(frame_indices, dtype=np.int64)] = relative_vals.astype(np.float32)

        absolute_values[0] = 0.0
        return absolute_values, relative_values


class Pi06RolloutEvaluator(BaseRolloutEvaluator):
    evaluator_model_type = "pi06"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name_or_path: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        attn_implementation: str = "flash_attention_2",
        bf16: bool = True,
        device: Optional[str] = None,
        max_new_tokens: int = 4,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        self.max_new_tokens = int(max_new_tokens)
        run_config = load_run_config(self.checkpoint_dir)
        base_model_name = model_name_or_path
        if base_model_name is None:
            base_model_name = (
                run_config.get("model_args", {}).get("model_name_or_path")
                or run_config.get("base_model_name_or_path")
            )
        processor_dir = find_processor_dir(self.checkpoint_dir)
        self.processor = AutoProcessor.from_pretrained(processor_dir)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        data_args_snapshot = run_config.get("data_args", {})
        if data_args_snapshot:
            self.processor = update_processor_pixels(self.processor, SimpleNamespace(**data_args_snapshot))

        self.model, self.model_arch = _load_qwen_generation_model(
            self.checkpoint_dir,
            attn_implementation=attn_implementation,
            bf16=bf16 and self.device.type == "cuda",
        )
        self.model.to(self.device)
        self.model.eval()

        tokenizer_ref = processor_dir if processor_dir.exists() else (base_model_name or str(self.checkpoint_dir))
        self.value_tokenizer = ValueTokenizer(
            llm_path=str(tokenizer_ref),
            bins=int(data_args_snapshot.get("value_tokenizer_bins", 201)),
            min_value=float(data_args_snapshot.get("value_tokenizer_min", -1.0)),
            max_value=float(data_args_snapshot.get("value_tokenizer_max", 0.0)),
        )
        if getattr(self.model, "generation_config", None) is not None and self.processor.tokenizer is not None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.processor.tokenizer.eos_token_id

        logger.info(
            "Pi06RolloutEvaluator ready: device=%s checkpoint=%s processor=%s arch=%s",
            self.device,
            self.checkpoint_dir,
            processor_dir,
            self.model_arch,
        )

    @staticmethod
    def _build_user_message(prompt: str, images: list[Image.Image]) -> list[dict[str, Any]]:
        image_pool = [{"type": "image", "image": image} for image in images]
        content = []
        for segment in re.split(r"(<image>)", VALUE_PROMPT_TEMPLATE.format(instruction=prompt)):
            if segment == "<image>":
                if not image_pool:
                    raise ValueError("Number of <image> placeholders exceeds provided images.")
                content.append(image_pool.pop(0))
            elif segment.strip():
                content.append({"type": "text", "text": segment.strip()})
        if image_pool:
            raise ValueError(f"{len(image_pool)} image(s) remain unused in pi06 sample.")
        return [{"role": "user", "content": content}]

    def _build_generation_instance(
        self,
        prompt: str,
        frame_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> dict[str, Any]:
        messages = self._build_user_message(prompt, self._frame_triplet_to_images(frame_triplet))
        instance = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(instance.get("input_ids"), list):
            instance["input_ids"] = torch.tensor(instance["input_ids"]).unsqueeze(0)
        if "attention_mask" not in instance:
            input_ids = instance["input_ids"]
            instance["attention_mask"] = input_ids.ne(self.processor.tokenizer.pad_token_id).long()
        return instance

    def _collate_generation_batch(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        attention_masks = [instance["attention_mask"].squeeze(0) for instance in instances]
        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_masks,
                batch_first=True,
                padding_value=0,
            ),
        }

        if any("token_type_ids" in instance for instance in instances):
            token_type_ids = [instance["token_type_ids"].squeeze(0) for instance in instances]
            batch["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                token_type_ids,
                batch_first=True,
                padding_value=0,
            )

        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "rope_deltas"):
            values = [instance[key] for instance in instances if key in instance]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch

    def _decode_generated_values(self, generated_token_ids: torch.Tensor) -> np.ndarray:
        values = []
        token_rows = generated_token_ids.detach().cpu().numpy()
        special_ids = {
            tid
            for tid in (
                self.processor.tokenizer.pad_token_id,
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.bos_token_id,
            )
            if tid is not None
        }

        for row in token_rows:
            value_token_id = None
            fallback_tokens = []
            for token_id in row.tolist():
                token_id = int(token_id)
                if token_id in self.value_tokenizer.token_id_to_bin_idx:
                    value_token_id = token_id
                    break
                if token_id not in special_ids:
                    fallback_tokens.append(token_id)

            if value_token_id is not None:
                decoded = self.value_tokenizer.decode_token_ids_to_values(
                    np.asarray([value_token_id], dtype=np.int64)
                )
                values.append(float(decoded[0]))
                continue

            decoded_text = self.processor.tokenizer.decode(fallback_tokens, skip_special_tokens=True).strip()
            matched = _FLOAT_PATTERN.search(decoded_text)
            if matched is None:
                raise ValueError(
                    "Could not decode pi06 value token from generation output. "
                    f"tokens={row.tolist()} text={decoded_text!r}"
                )
            values.append(float(matched.group(0)))

        return np.asarray(values, dtype=np.float32)

    def _prepare_generation_batch(
        self,
        top_frames: list[np.ndarray],
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        prompt: str,
        batch_start: int,
    ) -> tuple[list[int], dict[str, Any]]:
        num_frames = len(top_frames)
        batch_end = min(batch_start + self.batch_size, num_frames)
        frame_indices = list(range(batch_start, batch_end))
        instances = []
        for frame_idx in frame_indices:
            frame_triplet = (top_frames[frame_idx], left_frames[frame_idx], right_frames[frame_idx])
            instances.append(self._build_generation_instance(prompt=prompt, frame_triplet=frame_triplet))
        return frame_indices, self._collate_generation_batch(instances)

    def _predict_metric_arrays(
        self,
        top_frames: list[np.ndarray],
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        prompt: str,
        relative_interval: int,
        prefetch: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        del relative_interval
        num_frames = len(top_frames)
        batch_starts = list(range(0, num_frames, self.batch_size))
        absolute_values = np.zeros(num_frames, dtype=np.float32)

        def prepare(batch_start: int):
            return self._prepare_generation_batch(
                top_frames=top_frames,
                left_frames=left_frames,
                right_frames=right_frames,
                prompt=prompt,
                batch_start=batch_start,
            )

        prefetch_future = None
        if prefetch and batch_starts:
            prefetch_future = self._executor.submit(prepare, batch_starts[0])

        for batch_idx, batch_start in enumerate(batch_starts):
            if prefetch and prefetch_future is not None:
                frame_indices, batch = prefetch_future.result()
            else:
                frame_indices, batch = prepare(batch_start)

            if prefetch and batch_idx + 1 < len(batch_starts):
                prefetch_future = self._executor.submit(prepare, batch_starts[batch_idx + 1])

            batch = self._move_batch_to_device(batch, self.device)
            prompt_len = batch["input_ids"].shape[1]
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
            }
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                generated = self.model.generate(
                    **model_inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            new_tokens = generated.sequences[:, prompt_len:]
            batch_values = self._decode_generated_values(new_tokens)
            absolute_values[np.asarray(frame_indices, dtype=np.int64)] = batch_values

        return absolute_values, None


def build_rollout_evaluator(
    model_type: str,
    checkpoint_dir: str | Path,
    model_name_or_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 8,
    attn_implementation: str = "flash_attention_2",
    bf16: bool = True,
    device: Optional[str] = None,
):
    model_type = str(model_type).strip().lower()
    if model_type == "pair":
        return PairRolloutEvaluator(
            checkpoint_dir=checkpoint_dir,
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            num_workers=num_workers,
            attn_implementation=attn_implementation,
            bf16=bf16,
            device=device,
        )
    if model_type == "pi06":
        return Pi06RolloutEvaluator(
            checkpoint_dir=checkpoint_dir,
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            num_workers=num_workers,
            attn_implementation=attn_implementation,
            bf16=bf16,
            device=device,
        )
    raise ValueError(f"Unsupported model_type={model_type!r}. Expected one of: pair, pi06")


PairValueEvaluator = PairRolloutEvaluator
