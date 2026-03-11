import bisect
import os
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

VALUE_PROMPT_TEMPLATE = """ You are a rigorous, impartial vision evaluator for robot task progress. Given a task instruction and three-views observation images, your job is to estimate the current progress toward accomplishing the task.

# Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to Task Instruction.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
 - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
 - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
 - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
 - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.

Task Instruction: {instruction}
Robot Front Image: <image>
Robot Left Wrist Image: <image>
Robot Right Wrist Image: <image>
 Progress toward accomplishing the task: """

FRAME_CACHE_SIZE = 64
EPISODE_CACHE_SLOTS = 4


class MapPi06Dataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        seed: int = 42,
        val_ratio: float = 0.1,
        camera_names: Optional[list[str]] = None,
        value_tokenizer=None,
        split: str = "train",
        max_episodes: Optional[int] = None,
        language_instruction: str = "perform the task",
    ) -> None:
        super().__init__()
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.seed = int(seed)
        self.camera_names = camera_names or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        self.value_tokenizer = value_tokenizer
        self.split = split
        self.language_instruction = language_instruction
        self.penalty_ratio = 0.4

        self._lerobot_dataset = None
        self._init_pid = None

        self.episodes_meta = self._load_episodes_metadata(max_episodes=max_episodes)
        self.episodes_meta = self._split_train_val(self.episodes_meta, val_ratio, self.seed, split)
        self._global_step_prefix: list[int] = []
        self._build_indices()

        self._episode_caches: dict[int, OrderedDict[int, dict[str, Image.Image]]] = {}
        self._episode_cache_order: OrderedDict[int, None] = OrderedDict()

        print(
            f"[PI06-MAP-{split.upper()}] dataset={self.dataset_dir} "
            f"episodes={len(self.episodes_meta)} steps={len(self)}"
        )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_lerobot_dataset"] = None
        state["_init_pid"] = None
        state["_episode_caches"] = {}
        state["_episode_cache_order"] = OrderedDict()
        return state

    def _build_indices(self):
        total = 0
        for ep_info in self.episodes_meta:
            total += int(ep_info["length"])
            self._global_step_prefix.append(total)

    def __len__(self):
        return self._global_step_prefix[-1] if self._global_step_prefix else 0

    @staticmethod
    def _normalize_instruction(raw_instruction: Any, default_instruction: str) -> str:
        if isinstance(raw_instruction, list) and raw_instruction:
            raw_instruction = raw_instruction[0]
        if raw_instruction is None:
            return default_instruction
        return str(raw_instruction)

    def _load_episodes_metadata_from_dataset(self, max_episodes: Optional[int]) -> list[dict[str, Any]]:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(self.dataset_dir, video_backend="pyav")
        total_episodes = dataset.num_episodes
        if max_episodes is not None:
            total_episodes = min(total_episodes, int(max_episodes))

        episodes = []
        for ep_idx in range(total_episodes):
            start_index = int(dataset.episode_data_index["from"][ep_idx].item())
            end_index = int(dataset.episode_data_index["to"][ep_idx].item())
            instruction = self._normalize_instruction(
                dataset.meta.episodes[ep_idx].get("tasks"),
                self.language_instruction,
            )
            episodes.append(
                {
                    "episode_idx": ep_idx,
                    "global_start_index": start_index,
                    "length": end_index - start_index,
                    "instruction": instruction,
                    "success": True,
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
                instruction = self._normalize_instruction(ep_meta.get("tasks"), self.language_instruction)
                episodes.append(
                    {
                        "episode_idx": int(ep_key),
                        "global_start_index": global_start_index,
                        "length": length,
                        "instruction": instruction,
                        "success": True,
                    }
                )
                global_start_index += length

            if episodes:
                return episodes
        except Exception as exc:
            print(
                f"[PI06-MAP] metadata-only init unavailable for {self.dataset_dir}: "
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
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            except ImportError:
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

    def _calculate_R_value(self, current_step_idx, episode_length, is_final_step, episode_success=True):
        episode_success = True if episode_success is None else bool(episode_success)
        remaining_steps = episode_length - current_step_idx - 1
        penalty_ratio = self.penalty_ratio
        if is_final_step:
            R = 0.0 if episode_success else -(penalty_ratio * episode_length)
        else:
            current_reward = -1.0
            future_reward_val = 0.0 if episode_success else -(penalty_ratio * episode_length)
            future_rewards = remaining_steps * (-1.0) + future_reward_val
            R = current_reward + future_rewards

        min_R = -(episode_length - 1)
        max_R = 0.0
        if max_R != min_R:
            eps = 1e-7
            normalized_R = (R - min_R) / (max_R - min_R) * (1.0 - 2 * eps) + (-1.0 + eps)
        else:
            normalized_R = -1.0 + 1e-7

        return float(normalized_R)

    def _format_value_text(self, value: float) -> str:
        if self.value_tokenizer is not None:
            return str(self.value_tokenizer(np.array([value])))
        return str(value)

    def _build_value_conversation(self, instruction: str, value: float) -> list[dict[str, str]]:
        return [
            {"from": "human", "value": VALUE_PROMPT_TEMPLATE.format(instruction=instruction)},
            {"from": "gpt", "value": self._format_value_text(value)},
        ]

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_pos = bisect.bisect_right(self._global_step_prefix, index)
        ep_prev = 0 if ep_pos == 0 else self._global_step_prefix[ep_pos - 1]
        step = index - ep_prev
        return ep_pos, step

    def __getitem__(self, index: int):
        ep_pos, step = self._resolve_index(int(index))
        ep_info = self.episodes_meta[ep_pos]
        step_images = self._get_step_images(ep_info, step)
        image_list = [step_images[cam] for cam in self.camera_names]
        is_final = step == int(ep_info["length"]) - 1
        value = self._calculate_R_value(step, int(ep_info["length"]), is_final, ep_info.get("success", True))
        return {
            "conversations": self._build_value_conversation(ep_info["instruction"], value),
            "image": image_list,
            "meta_value": float(value),
            "meta_ep_idx": int(ep_info["episode_idx"]),
            "meta_t": int(step),
            "meta_episode_len": int(ep_info["length"]),
            "meta_dataset_dir": self.dataset_dir,
        }
