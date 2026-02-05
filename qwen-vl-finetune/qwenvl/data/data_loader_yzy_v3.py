import math
import traceback
from typing import Optional, List, Dict, Any
import os
import numpy as np
from PIL import Image

from torch.utils.data import IterableDataset, get_worker_info
import torch

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

class LeRobotValueDataset(IterableDataset):
    """
    Dataset class optimized for Qwen2.5-VL with LeRobot v2.1 format data using Shuffle Buffer strategy.

    For each episode, provides:
    1. Language instruction: Task name from dataset
    2. Observation: Images from observation.images.{camera_name}
    3. Value (R): Computed cumulative reward from current step to end

    Uses IterableDataset + Shuffle Buffer strategy:
    1. Worker sharding: Different GPU/Workers process different episodes
    2. Sequential reading: For IO efficiency (especially video mode), reads episode frames sequentially
    3. Shuffle Buffer: Frames are put into buffer and shuffled, only yields when buffer is full to break temporal correlation

    Dataset structure:
    - LeRobot v2.1 format with episodes and frames
    - Features: observation.state, action, observation.images.{cam_name}

    Each episode corresponds to one trajectory.
    """

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
        value_tokenizer = None,
        split: str = "train", # "train" or "val"
        max_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.language_instruction = language_instruction
        self.seed = seed
        self.buffer_size = buffer_size
        self.value_tokenizer = value_tokenizer
        self.camera_names = camera_names or ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        self.split = split
        # Trainer/Distributed can call set_epoch() to reshuffle each epoch.
        self._epoch = 0
        self.penalty_ratio = 0.4
        # IMPORTANT:
        # - LeRobotDataset(video_backend="pyav") 会触发底层 ffmpeg/pyav 资源初始化；
        # - 在 DataLoader 多进程 (尤其是默认 fork) 下，父进程持有这些对象会导致 worker 退出时 SIGABRT：
        #   "terminate called without an active exception" / "worker is killed by signal: Aborted"
        # 因此这里仅用一个临时实例读取 metadata，然后丢弃；真正的逐帧读取在每个 worker 内部懒加载。
        self._lerobot_dataset = None
        self._init_pid = None

        tmp_ds = LeRobotDataset(dataset_dir, video_backend="pyav")

        # Extract episode_data_index into plain python lists (picklable / fork-safe)
        from_indices = tmp_ds.episode_data_index["from"]
        to_indices = tmp_ds.episode_data_index["to"]
        self._episode_from = [int(x) for x in from_indices.tolist()]
        self._episode_to = [int(x) for x in to_indices.tolist()]

        # Calculate global max episode length for R value normalization BEFORE filtering episodes
        if len(self._episode_from) > 0:
            all_lengths = [t - f for f, t in zip(self._episode_from, self._episode_to)]
            max_len = max(all_lengths)
            min_len = min(all_lengths)
        else:
            max_len = 0
            min_len = 0
        print(f"[DEBUG] max_len calculated: {max_len}, all_lengths range: [{min_len}, {max_len}]")
        
        self.global_min_R = -(max_len - 1 + self.penalty_ratio * max_len)
        self.global_max_R = 0.0

        # Extract episode information from metadata efficiently (without loading images)
        self.episodes_meta = self._load_episodes_metadata(tmp_ds, max_episodes)
        # print(f"[DEBUG] Loaded {len(self.episodes_meta)} episodes metadata")

        # Split into train/validation sets
        self.episodes_meta = self._split_train_val(self.episodes_meta, val_ratio, seed, split)
        # print(f"[DEBUG] Split complete, {len(self.episodes_meta)} episodes after split")

        # Drop temporary dataset to avoid forking C++/ffmpeg state
        try:
            del tmp_ds
        except Exception:
            pass

        print(f"[{split.upper()}] Dataset initialized. Episodes: {len(self.episodes_meta)}. "
              f"Global R range: [{self.global_min_R}, {self.global_max_R}]")

    def set_epoch(self, epoch: int) -> None:
        """
        Let external trainers (e.g., HF Trainer / Accelerate) set the current epoch.
        We will incorporate epoch into RNG seed so episode order changes each epoch.
        """
        try:
            self._epoch = int(epoch)
        except Exception:
            self._epoch = 0

    def _load_episodes_metadata(self, lerobot_dataset, max_episodes):
        """Load only LeRobot metadata information."""
        meta = lerobot_dataset.meta
        total_episodes = lerobot_dataset.num_episodes
        # print(f"[DEBUG] _load_episodes_metadata: total_episodes={total_episodes}, max_episodes={max_episodes}")
        episodes = []

        # Limit maximum number of episodes
        if max_episodes is not None:
            total_episodes = min(total_episodes, max_episodes)

        # print(f"[DEBUG] Starting to iterate through {total_episodes} episodes...")
        for ep_idx in range(total_episodes):
            # Get global start and end indices in parquet/hf_dataset
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
                'success': bool(lerobot_dataset[start_index]['result'])  # Assume all LeRobot episodes are successful demonstrations
            })

            # print(f"[DEBUG] Episode {ep_idx} metadata: {episodes[-1]}")
        
        # print(f"[DEBUG] Completed loading {len(episodes)} episodes metadata")
        return episodes

    def _get_lerobot_dataset(self):
        """
        Lazily create LeRobotDataset inside the current process (main or DataLoader worker).
        This avoids inheriting ffmpeg/pyav state across fork.
        """
        pid = os.getpid()
        if self._lerobot_dataset is None or self._init_pid != pid:
            self._lerobot_dataset = LeRobotDataset(self.dataset_dir, video_backend="pyav")
            self._init_pid = pid
        return self._lerobot_dataset

    def __getstate__(self):
        """
        Ensure dataset object is not pickled (for spawn) and not copied with C++ state.
        """
        state = dict(self.__dict__)
        state["_lerobot_dataset"] = None
        state["_init_pid"] = None
        return state

    def _split_train_val(self, episodes, val_ratio, seed, split):
        """Split episodes into train/validation sets."""
        n = len(episodes)
        if n == 0: return []
        
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        
        n_val = int(n * val_ratio)
        if split == "val":
            selected_indices = indices[:n_val]
        else:
            selected_indices = indices[n_val:]
            
        return [episodes[i] for i in selected_indices]

    def _calculate_R_value(self, current_step_idx, episode_length, is_final_step, episode_success=True):
        """Calculate R value for value function training."""
        remaining_steps = episode_length - current_step_idx - 1
        penalty_ratio = self.penalty_ratio
        if is_final_step:
            R = 0.0 if episode_success else -(penalty_ratio * episode_length)
        else:
            current_reward = -1.0
            future_reward_val = 0.0 if episode_success else -(penalty_ratio * episode_length)
            future_rewards = remaining_steps * (-1.0) + future_reward_val
            R = current_reward + future_rewards
            if episode_success:
                R = -self.global_min_R/(episode_length - 1) * R

        # Normalize
        #min_R = -(episode_length - 1) if episode_success else - math.floor((1+penalty_ratio) * episode_length - 1)
        min_R = self.global_min_R
        max_R = self.global_max_R
        
        if max_R != min_R:
            eps = 1e-7
            normalized_R = (R - min_R) / (max_R - min_R) * (1.0 - 2*eps) + (-1.0 + eps)
        else:
            normalized_R = -1.0 + 1e-7

        return normalized_R

    def _process_frame_data(self, raw_row, episode_info, step):
        """Process single frame data: convert image format and calculate value.

        Strictly follows the output format from robotwin_to_lerobot.py:
        - Images: numpy array, shape (3, 224, 224), dtype uint8
        - Stored in observation.images.{cam} fields
        """
        # Process images
        image_list = []
        for cam in self.camera_names:
            key = f"observation.images.{cam}"
            if key not in raw_row:
                raise ValueError(f"Missing required camera data: {key} in episode {episode_info['episode_idx']}, step {step}")

            img_data = raw_row[key]

            # Convert format: (C, H, W) -> (H, W, C) -> PIL Image
            img_rgb = img_data.permute(1, 2, 0)  # (3, 224, 224) -> (224, 224, 3)
            img_rgb_np = img_rgb.numpy()

            # Convert float32 [0,1] to uint8 [0,255] for PIL compatibility
            if img_rgb_np.dtype == np.float32:
                img_rgb_np = (img_rgb_np * 255).astype(np.uint8)

            image_list.append(Image.fromarray(img_rgb_np, mode='RGB'))

        # Calculate value
        is_final = (step == episode_info['length'] - 1)
        R_val = self._calculate_R_value(step, episode_info['length'], is_final, episode_info['success'])

        if self.value_tokenizer:
            value_str = self.value_tokenizer(np.array([R_val]))
        else:
            value_str = str(R_val)

        # Construct Qwen format
        conversation = [
            {
                "from": "human",
                "value": f""" You are a rigorous, impartial vision evaluator for robot task progress. Given a task instruction and three-views observation images, your job is to estimate the current progress toward accomplishing the task.\n\n# Evaluation Criteria (apply across all three views)\n1) Task Alignment: Evidence directly tied to Task Instruction.\n2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.\n3) View-Specific Evidence & Consistency:\n - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.\n - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).\n - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.\n - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.\n4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.\n\nTask Instruction: {episode_info['instruction']}\nRobot Front Image: <image>\nRobot Left Wrist Image: <image>\nRobot Right Wrist Image: <image>\n Progress toward accomplishing the task: """,
            },
            {
                "from": "gpt",
                "value": value_str,
            }
        ]

        return {
            "conversations": conversation,
            "image": image_list,
            # Debug info
            "meta_R": R_val,
            "meta_ep_idx": episode_info['episode_idx']
        }

    def __iter__(self):
        """
        Core iteration logic:
        (Distributed rank + DataLoader worker) sharding
        -> Episode shuffling (epoch-aware)
        -> Frame sequential read
        -> Shuffle Buffer -> Yield

        Notes:
        - We shard by episode (not by frame) to preserve sequential access within each episode.
        - We *do not* mutate self.episodes_meta in-place (important for reproducibility/debugging).
        """
        worker_info = get_worker_info()

        # --- Resolve distributed rank/world_size (if any) ---
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
            world_size = int(torch.distributed.get_world_size())
        else:
            rank = 0
            world_size = 1

        # --- Resolve dataloader worker id/num_workers (if any) ---
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        # Global worker id across all ranks (avoid multi-GPU duplication)
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        # --- Determine episodes assigned to this (rank, worker) ---
        # Shard by episode index stride to keep it simple and stable.
        full_list = self.episodes_meta
        my_episodes = [full_list[i] for i in range(global_worker_id, len(full_list), global_num_workers)]

        # --- Shuffle episode processing order (epoch-aware) ---
        # Use a deterministic seed per epoch & global worker to reshuffle each epoch.
        # This does NOT affect train/val split (that is done in __init__).
        epoch = getattr(self, "_epoch", 0) or 0
        rng = np.random.default_rng(self.seed + epoch * 1000 + global_worker_id)
        rng.shuffle(my_episodes)

        buffer = []
        
        # Iterate through episodes assigned to this worker
        for ep_info in my_episodes:
            try:
                # Optimization: read directly from hf_dataset range to avoid repeated __getitem__
                # LeRobot uses Arrow/Parquet underneath, range reading is typically more efficient
                # Note: can fall back to row-by-row reading if memory is insufficient
                start = ep_info['global_start_index']
                length = ep_info['length']

                # Process frame by frame (sequential access within episode)
                for step in range(length):
                    global_idx = start + step

                    # Key fix: access underlying dataset directly, bypassing LeRobot wrapper
                    raw_row = self._get_lerobot_dataset()[global_idx]

                    # Process data
                    processed_sample = self._process_frame_data(raw_row, ep_info, step)
                    
                    # Add to shuffle buffer
                    if len(buffer) < self.buffer_size:
                        buffer.append(processed_sample)
                    else:
                        pop_idx = rng.integers(0, len(buffer))
                        yield buffer[pop_idx]
                        buffer[pop_idx] = processed_sample

            except Exception as e:
                print(f"Error reading episode {ep_info['episode_idx']}: {e}")
                traceback.print_exc()
                continue

        # Flush buffer
        rng.shuffle(buffer)
        for item in buffer:
            yield item

    def __len__(self):
        total_frames = sum(e['length'] for e in self.episodes_meta)
        return total_frames