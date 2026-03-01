import math
import traceback
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

from torch.utils.data import IterableDataset, get_worker_info

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

PAIR_PROMPT_TEMPLATE = """You are a progress estimator for robot manipulation tasks.
Given two sets of three-view observations at time t0 and time t1, estimate the signed progress change from t0 to t1.

Task Instruction: {instruction}

Observation at t0:
Top view: <image>
Left wrist view: <image>
Right wrist view: <image>

Observation at t1:
Top view: <image>
Left wrist view: <image>
Right wrist view: <image>

Progress delta from t0 to t1: """


class LeRobotPairDataset(IterableDataset):
    def __init__(
        self,
        dataset_dir: str,
        transform=None,
        tokenizer=None,
        language_instruction: str = "perform the task",
        seed: int = 42,
        val_ratio: float = 0.1,
        buffer_size: int = 500,
        camera_names: Optional[List[str]] = None,
        split: str = "train",
        max_episodes: Optional[int] = None,
        pair_short_step: int = 8,
        pair_mid_step: int = 16,
        pair_random_min: int = 1,
        pair_add_backward: bool = True,
        pair_prompt_style: str = "explicit_t0_t1",
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.language_instruction = language_instruction
        self.seed = seed
        self.buffer_size = buffer_size
        self.camera_names = camera_names or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        self.split = split

        self.pair_short_step = int(pair_short_step)
        self.pair_mid_step = int(pair_mid_step)
        self.pair_random_min = max(1, int(pair_random_min))
        self.pair_add_backward = bool(pair_add_backward)
        self.pair_prompt_style = pair_prompt_style

        self.lerobot_dataset = LeRobotDataset(dataset_dir, video_backend="pyav")
        self.episodes_meta = self._load_episodes_metadata(max_episodes)
        self.episodes_meta = self._split_train_val(self.episodes_meta, val_ratio, seed, split)

        print(
            f"[{split.upper()}] Pair dataset initialized. Episodes: {len(self.episodes_meta)}. "
            f"short={self.pair_short_step}, mid={self.pair_mid_step}, random_min={self.pair_random_min}, "
            f"add_backward={self.pair_add_backward}"
        )

    def _load_episodes_metadata(self, lerobot_dataset, max_episodes):
        """Load only LeRobot metadata information."""
        total_episodes = lerobot_dataset.num_episodes
        episodes = []

        if max_episodes is not None:
            total_episodes = min(total_episodes, max_episodes)

        for ep_idx in range(total_episodes):
            start_index = int(lerobot_dataset.episode_data_index["from"][ep_idx].item())
            end_index = int(lerobot_dataset.episode_data_index["to"][ep_idx].item())
            length = end_index - start_index

            instruction = lerobot_dataset.meta.episodes[ep_idx].get("tasks")
            if isinstance(instruction, list) and len(instruction) > 0:
                instruction = str(instruction[0])

            episodes.append({
                'episode_idx': ep_idx,
                'global_start_index': start_index,
                'length': length,
                'instruction': instruction,
            })

        return episodes

    def _split_train_val(self, episodes, val_ratio, seed, split):
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

    def _build_pair_conversation(self, instruction: str) -> List[Dict[str, str]]:
        if self.pair_prompt_style == "explicit_t0_t1":
            prompt = PAIR_PROMPT_TEMPLATE.format(instruction=instruction)
        else:
            prompt = PAIR_PROMPT_TEMPLATE.format(instruction=instruction)
        return [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": ""},
        ]

    def _construct_pairs_for_t(self, t: int, T: int, rng: np.random.Generator):
        forward_pairs = []
        forward_pairs.append((0, t, float(t / T), "anchor"))
        if t >= self.pair_short_step:
            forward_pairs.append((t - self.pair_short_step, t, float(self.pair_short_step / T), "short"))
        if t >= self.pair_mid_step:
            forward_pairs.append((t - self.pair_mid_step, t, float(self.pair_mid_step / T), "mid"))
        forward_pairs.append((t, t, 0.0, "zero"))
        if t >= self.pair_random_min:
            r = int(rng.integers(self.pair_random_min, t + 1))
            forward_pairs.append((t - r, t, float(r / T), "random"))
        return forward_pairs

    def _process_pair_data(
        self,
        frame_cache: Dict[int, Dict[str, Image.Image]],
        ep_info: Dict[str, Any],
        t: int,
        a: int,
        b: int,
        delta: float,
        pair_type: str,
        t_group_weight: float,
    ):
        image_list = []
        for cam in self.camera_names:
            image_list.append(frame_cache[a][cam])
        for cam in self.camera_names:
            image_list.append(frame_cache[b][cam])

        return {
            "conversations": self._build_pair_conversation(ep_info["instruction"]),
            "image": image_list,
            "delta_label": float(np.clip(delta, -1.0, 1.0)),
            "pair_type": pair_type,
            "t_group_weight": float(t_group_weight),
            "meta_ep_idx": int(ep_info["episode_idx"]),
            "meta_t": int(t),
            "meta_a": int(a),
            "meta_b": int(b),
            "meta_episode_len": int(ep_info["length"]),
        }

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            my_episodes = list(self.episodes_meta)
            worker_id = 0
        else:
            full_list = self.episodes_meta
            per_worker = int(math.ceil(len(full_list) / float(worker_info.num_workers)))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, len(full_list))
            my_episodes = list(full_list[start_idx:end_idx])
            worker_id = worker_info.id

        rng = np.random.default_rng(self.seed + worker_id)
        rng.shuffle(my_episodes)
        buffer = []

        for ep_info in my_episodes:
            try:
                start = ep_info["global_start_index"]
                length = ep_info["length"]
                if length <= 0:
                    continue

                frame_cache: Dict[int, Dict[str, Image.Image]] = {}
                for step in range(length):
                    raw_row = self.lerobot_dataset[start + step]
                    frame_cache[step] = {}
                    for cam in self.camera_names:
                        key = f"observation.images.{cam}"
                        if key not in raw_row:
                            raise ValueError(
                                f"Missing required camera data: {key} in episode {ep_info['episode_idx']} step {step}"
                            )
                        frame_cache[step][cam] = self._to_pil_image(raw_row[key])

                ep_rng = np.random.default_rng(self.seed + ep_info["episode_idx"] * 9973 + worker_id)

                for t in range(length):
                    forward_pairs = self._construct_pairs_for_t(t, length, ep_rng)
                    n_pairs_at_t = len(forward_pairs) * (2 if self.pair_add_backward else 1)
                    t_group_weight = 1.0 / max(1, n_pairs_at_t)

                    for (a, b, delta, ptype) in forward_pairs:
                        sample = self._process_pair_data(
                            frame_cache, ep_info, t, a, b, delta, f"{ptype}_fwd", t_group_weight
                        )
                        if len(buffer) < self.buffer_size:
                            buffer.append(sample)
                        else:
                            pop_idx = int(rng.integers(0, len(buffer)))
                            yield buffer[pop_idx]
                            buffer[pop_idx] = sample

                        if self.pair_add_backward:
                            if not (a == b and abs(delta) < 1e-8):
                                back_sample = self._process_pair_data(
                                    frame_cache, ep_info, t, b, a, -delta, f"{ptype}_bwd", t_group_weight
                                )
                                if len(buffer) < self.buffer_size:
                                    buffer.append(back_sample)
                                else:
                                    pop_idx = int(rng.integers(0, len(buffer)))
                                    yield buffer[pop_idx]
                                    buffer[pop_idx] = back_sample

                del frame_cache
            except Exception as e:
                print(f"Error reading episode {ep_info['episode_idx']}: {e}")
                traceback.print_exc()
                continue

        rng.shuffle(buffer)
        for item in buffer:
            yield item

    def __len__(self):
        total = 0
        for ep in self.episodes_meta:
            T = ep["length"]
            for t in range(T):
                n = 2  # anchor + zero
                if t >= self.pair_short_step:
                    n += 1
                if t >= self.pair_mid_step:
                    n += 1
                if t >= self.pair_random_min:
                    n += 1
                if self.pair_add_backward:
                    n *= 2
                total += n
        return total
