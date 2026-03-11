import bisect
import os
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

PAIR_PROMPT_TEMPLATE = """You are estimating robot task progress change from paired observations.
Compare the current observation against the historical observation for the same task.

Task: {instruction}

Current observation:
Front view: <image>
Left wrist view: <image>
Right wrist view: <image>

Historical observation:
Front view: <image>
Left wrist view: <image>
Right wrist view: <image>

Progress change from historical to current:"""

FRAME_CACHE_SIZE = 64
EPISODE_CACHE_SLOTS = 4


class MapPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        seed: int = 42,
        val_ratio: float = 0.1,
        camera_names: Optional[list[str]] = None,
        split: str = "train",
        max_episodes: Optional[int] = None,
        pair_add_backward: bool = True,
        pair_add_zero_anchor: Optional[bool] = None,
        pair_target_column: Optional[str] = None,
        pair_prompt_style: str = "current_history_delta",
    ) -> None:
        super().__init__()
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.seed = int(seed)
        self.camera_names = camera_names or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        self.split = split
        self.max_episodes = max_episodes
        self.pair_add_backward = bool(pair_add_backward)
        self.pair_add_zero_anchor = (
            False if pair_add_zero_anchor is None else bool(pair_add_zero_anchor)
        )
        self.pair_target_column = str(pair_target_column).strip() if pair_target_column else None
        self.pair_prompt_style = pair_prompt_style

        self._lerobot_dataset = None
        self._init_pid = None

        self.episodes_meta = self._load_episodes_metadata(max_episodes=max_episodes)
        self.episodes_meta = self._split_train_val(self.episodes_meta, val_ratio, self.seed, split)
        self._episode_pair_prefix: list[list[int]] = []
        self._global_pair_prefix: list[int] = []
        self._build_indices()

        self._episode_caches: dict[int, OrderedDict[int, dict[str, Image.Image]]] = {}
        self._episode_target_caches: dict[int, OrderedDict[int, float]] = {}
        self._episode_cache_order: OrderedDict[int, None] = OrderedDict()

        print(
            f"[PAIR-MAP-{split.upper()}] dataset={self.dataset_dir} "
            f"episodes={len(self.episodes_meta)} pairs={len(self)} add_backward={self.pair_add_backward} "
            f"add_zero_anchor={self.pair_add_zero_anchor} target_column={self.pair_target_column}"
        )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_lerobot_dataset"] = None
        state["_init_pid"] = None
        state["_episode_caches"] = {}
        state["_episode_target_caches"] = {}
        state["_episode_cache_order"] = OrderedDict()
        return state

    @staticmethod
    def _normalize_instruction(raw_instruction: Any) -> str:
        if isinstance(raw_instruction, list) and raw_instruction:
            raw_instruction = raw_instruction[0]
        if raw_instruction is None:
            return "perform the task"
        return str(raw_instruction)

    def _load_episodes_metadata_from_dataset(self, max_episodes: Optional[int]) -> list[dict[str, Any]]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(self.dataset_dir, video_backend="pyav")
        total_episodes = dataset.num_episodes
        if max_episodes is not None:
            total_episodes = min(total_episodes, int(max_episodes))

        episodes = []
        for ep_idx in range(total_episodes):
            start_index = int(dataset.episode_data_index["from"][ep_idx].item())
            end_index = int(dataset.episode_data_index["to"][ep_idx].item())
            instruction = self._normalize_instruction(dataset.meta.episodes[ep_idx].get("tasks"))
            episodes.append(
                {
                    "episode_idx": ep_idx,
                    "global_start_index": start_index,
                    "length": end_index - start_index,
                    "instruction": instruction,
                }
            )
        return episodes

    def _load_episodes_metadata(self, max_episodes: Optional[int]) -> list[dict[str, Any]]:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

            meta = LeRobotDatasetMetadata(self.dataset_dir)
            raw_episodes = getattr(meta, "episodes", None)
            if not raw_episodes:
                raise ValueError("LeRobotDatasetMetadata.episodes is empty.")

            if isinstance(raw_episodes, dict):
                ordered_episode_items = sorted(raw_episodes.items(), key=lambda kv: int(kv[0]))
            else:
                ordered_episode_items = list(enumerate(raw_episodes))

            if max_episodes is not None:
                ordered_episode_items = ordered_episode_items[: int(max_episodes)]

            episodes = []
            global_start_index = 0
            for ep_key, ep_meta in ordered_episode_items:
                length = int(ep_meta["length"])
                instruction = self._normalize_instruction(ep_meta.get("tasks"))
                episodes.append(
                    {
                        "episode_idx": int(ep_key),
                        "global_start_index": global_start_index,
                        "length": length,
                        "instruction": instruction,
                    }
                )
                global_start_index += length

            if episodes:
                return episodes
        except Exception as exc:
            print(
                f"[PAIR-MAP] metadata-only init unavailable for {self.dataset_dir}: "
                f"{type(exc).__name__}: {exc}. Falling back to LeRobotDataset init."
            )

        return self._load_episodes_metadata_from_dataset(max_episodes=max_episodes)

    @staticmethod
    def _split_train_val(episodes, val_ratio, seed, split):
        n = len(episodes)
        if n == 0:
            return []
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        n_val = int(n * val_ratio)
        if n >= 2:
            n_val = max(1, min(n - 1, n_val))
        else:
            n_val = min(n, n_val)
        selected = indices[:n_val] if split == "val" else indices[n_val:]
        return [episodes[i] for i in selected]

    def _pair_count_for_t(self, t: int) -> int:
        if t <= 0:
            return 0
        count = 2 if self.pair_add_backward else 1
        if self.pair_add_zero_anchor:
            count += 2 if self.pair_add_backward else 1
        return count

    def _build_indices(self):
        total = 0
        for ep in self.episodes_meta:
            prefix = [0]
            running = 0
            for t in range(int(ep["length"])):
                running += self._pair_count_for_t(t)
                prefix.append(running)
            self._episode_pair_prefix.append(prefix)
            total += running
            self._global_pair_prefix.append(total)

    def __len__(self):
        return self._global_pair_prefix[-1] if self._global_pair_prefix else 0

    @staticmethod
    def _to_pil_image(img_data):
        img_rgb = img_data.permute(1, 2, 0)
        img_rgb_np = img_rgb.numpy()
        if img_rgb_np.dtype == np.float32:
            img_rgb_np = np.clip(img_rgb_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_rgb_np, mode="RGB")

    def _get_lerobot_dataset(self):
        pid = os.getpid()
        if self._lerobot_dataset is None or self._init_pid != pid:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._lerobot_dataset = LeRobotDataset(self.dataset_dir, video_backend="pyav")
            self._init_pid = pid
        return self._lerobot_dataset

    def _touch_episode_cache(self, ep_idx: int):
        if ep_idx in self._episode_cache_order:
            self._episode_cache_order.move_to_end(ep_idx)
            return
        self._episode_cache_order[ep_idx] = None
        if len(self._episode_cache_order) > EPISODE_CACHE_SLOTS:
            old_ep_idx, _ = self._episode_cache_order.popitem(last=False)
            self._episode_caches.pop(old_ep_idx, None)
            self._episode_target_caches.pop(old_ep_idx, None)

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected scalar target value, got sequence={value}")
            return MapPairDataset._to_scalar(value[0])
        return float(value)

    def _get_step_images(self, ep_info: dict[str, Any], step: int) -> dict[str, Image.Image]:
        ep_idx = int(ep_info["episode_idx"])
        self._touch_episode_cache(ep_idx)
        frame_cache = self._episode_caches.setdefault(ep_idx, OrderedDict())
        if step in frame_cache:
            frame_cache.move_to_end(step)
            return frame_cache[step]

        raw_row = self._get_lerobot_dataset()[int(ep_info["global_start_index"]) + int(step)]
        step_images: dict[str, Image.Image] = {}
        for cam in self.camera_names:
            key = f"observation.images.{cam}"
            if key not in raw_row:
                raise ValueError(
                    f"Missing required camera data: {key} in dataset={self.dataset_dir}, "
                    f"episode={ep_idx}, step={step}"
                )
            step_images[cam] = self._to_pil_image(raw_row[key])

        frame_cache[step] = step_images
        if len(frame_cache) > FRAME_CACHE_SIZE:
            frame_cache.popitem(last=False)
        return step_images

    def _get_step_target_value(self, ep_info: dict[str, Any], step: int) -> float:
        if not self.pair_target_column:
            raise RuntimeError("pair_target_column is not configured.")
        ep_idx = int(ep_info["episode_idx"])
        self._touch_episode_cache(ep_idx)
        target_cache = self._episode_target_caches.setdefault(ep_idx, OrderedDict())
        if step in target_cache:
            target_cache.move_to_end(step)
            return float(target_cache[step])

        raw_row = self._get_lerobot_dataset()[int(ep_info["global_start_index"]) + int(step)]
        if self.pair_target_column not in raw_row:
            raise KeyError(
                f"Missing pair target column {self.pair_target_column!r} in dataset={self.dataset_dir}, "
                f"episode={ep_idx}, step={step}"
            )
        target_value = self._to_scalar(raw_row[self.pair_target_column])
        target_cache[step] = float(target_value)
        if len(target_cache) > FRAME_CACHE_SIZE:
            target_cache.popitem(last=False)
        return float(target_value)

    def _build_pair_conversation(self, instruction: str) -> list[dict[str, str]]:
        prompt = PAIR_PROMPT_TEMPLATE.format(instruction=instruction)
        return [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]

    def _sample_history_step(self, episode_idx: int, current_step: int) -> int:
        if current_step <= 0:
            raise ValueError(f"current_step must be > 0, got {current_step}")
        rng = np.random.default_rng(self.seed + int(episode_idx) * 9973 + int(current_step) * 101)
        return int(rng.integers(0, current_step))

    def _variants_for_t(self, ep_info: dict[str, Any], t: int) -> list[tuple[int, int, float, str]]:
        if t <= 0:
            return []
        out = []
        history_step = self._sample_history_step(int(ep_info["episode_idx"]), t)
        if self.pair_target_column:
            delta = float(
                np.clip(
                    self._get_step_target_value(ep_info, t) - self._get_step_target_value(ep_info, history_step),
                    -1.0,
                    1.0,
                )
            )
        else:
            episode_length = max(1, int(ep_info["length"]))
            delta = float(np.clip((t - history_step) / episode_length, -1.0, 1.0))
        out.append((t, history_step, delta, "current_history_fwd"))
        if self.pair_add_backward:
            out.append((history_step, t, -delta, "current_history_bwd"))

        if self.pair_add_zero_anchor:
            if self.pair_target_column:
                zero_delta = float(
                    np.clip(
                        self._get_step_target_value(ep_info, t) - self._get_step_target_value(ep_info, 0),
                        -1.0,
                        1.0,
                    )
                )
            else:
                episode_length = max(1, int(ep_info["length"]))
                zero_delta = float(np.clip(t / episode_length, -1.0, 1.0))
            out.append((t, 0, zero_delta, "current_zero_fwd"))
            if self.pair_add_backward:
                out.append((0, t, -zero_delta, "current_zero_bwd"))
        return out

    def _resolve_index(self, index: int) -> tuple[int, int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_pos = bisect.bisect_right(self._global_pair_prefix, index)
        ep_prev = 0 if ep_pos == 0 else self._global_pair_prefix[ep_pos - 1]
        local_pair_idx = index - ep_prev
        t_prefix = self._episode_pair_prefix[ep_pos]
        t = bisect.bisect_right(t_prefix, local_pair_idx) - 1
        pair_idx_at_t = local_pair_idx - t_prefix[t]
        return ep_pos, t, pair_idx_at_t

    def __getitem__(self, index: int):
        ep_pos, t, pair_idx = self._resolve_index(int(index))
        ep_info = self.episodes_meta[ep_pos]
        variants = self._variants_for_t(ep_info, t)
        if not variants:
            raise RuntimeError(
                f"Resolved pair index points to an empty timestep: dataset={self.dataset_dir}, "
                f"episode={ep_info['episode_idx']}, t={t}"
            )

        current_step, history_step, delta, pair_type = variants[pair_idx]
        current_images = self._get_step_images(ep_info, current_step)
        history_images = self._get_step_images(ep_info, history_step)

        image_list = []
        for cam in self.camera_names:
            image_list.append(current_images[cam])
        for cam in self.camera_names:
            image_list.append(history_images[cam])

        t_group_weight = 1.0 / len(variants)
        return {
            "conversations": self._build_pair_conversation(ep_info["instruction"]),
            "image": image_list,
            "delta_label": float(np.clip(delta, -1.0, 1.0)),
            "pair_type": pair_type,
            "t_group_weight": float(t_group_weight),
            "meta_ep_idx": int(ep_info["episode_idx"]),
            "meta_t": int(t),
            "meta_current": int(current_step),
            "meta_history": int(history_step),
            "meta_episode_len": int(ep_info["length"]),
            "meta_dataset_dir": self.dataset_dir,
        }
