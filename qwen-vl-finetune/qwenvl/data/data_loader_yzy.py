import math
import traceback
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

from torch.utils.data import IterableDataset, get_worker_info

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

        # Load dataset (metadata only)
        # print(f"[DEBUG] Loading LeRobotDataset from {dataset_dir}...")
        self.lerobot_dataset = LeRobotDataset(dataset_dir, video_backend="pyav")
        # print(f"[DEBUG] LeRobotDataset loaded, accessing episode_data_index...")

        # Calculate global max episode length for R value normalization BEFORE filtering episodes
        # Use efficient tensor operations on metadata
        from_indices = self.lerobot_dataset.episode_data_index["from"]
        to_indices = self.lerobot_dataset.episode_data_index["to"]
        # print(f"[DEBUG] episode_data_index accessed. Total episodes: {len(from_indices)}")
        
        all_lengths = to_indices - from_indices
        max_len = all_lengths.max().item() if len(all_lengths) > 0 else 0
        print(f"[DEBUG] max_len calculated: {max_len}, all_lengths range: [{all_lengths.min().item()}, {all_lengths.max().item()}]")
        
        self.global_min_R = -(max_len - 1 + 0.1 * max_len)
        self.global_max_R = 0.0

        # Extract episode information from metadata efficiently (without loading images)
        self.episodes_meta = self._load_episodes_metadata(max_episodes)
        # print(f"[DEBUG] Loaded {len(self.episodes_meta)} episodes metadata")

        # Split into train/validation sets
        self.episodes_meta = self._split_train_val(self.episodes_meta, val_ratio, seed, split)
        # print(f"[DEBUG] Split complete, {len(self.episodes_meta)} episodes after split")

        print(f"[{split.upper()}] Dataset initialized. Episodes: {len(self.episodes_meta)}. "
              f"Global R range: [{self.global_min_R}, {self.global_max_R}]")

    def _load_episodes_metadata(self, max_episodes):
        """Load only LeRobot metadata information."""
        meta = self.lerobot_dataset.meta
        total_episodes = self.lerobot_dataset.num_episodes
        # print(f"[DEBUG] _load_episodes_metadata: total_episodes={total_episodes}, max_episodes={max_episodes}")
        episodes = []

        # Limit maximum number of episodes
        if max_episodes is not None:
            total_episodes = min(total_episodes, max_episodes)

        # print(f"[DEBUG] Starting to iterate through {total_episodes} episodes...")
        for ep_idx in range(total_episodes):
            # Get global start and end indices in parquet/hf_dataset
            start_index = self.lerobot_dataset.episode_data_index["from"][ep_idx].item()
            end_index = self.lerobot_dataset.episode_data_index["to"][ep_idx].item()
            length = end_index - start_index

            instruction = self.lerobot_dataset.meta.episodes[ep_idx].get("tasks")
            if isinstance(instruction, list) and len(instruction) > 0:
                instruction = str(instruction[0])

            episodes.append({
                'episode_idx': ep_idx,
                'global_start_index': start_index,
                'length': length,
                'instruction': instruction,
                'success': bool(self.lerobot_dataset[start_index]['result'])  # Assume all LeRobot episodes are successful demonstrations
            })

            # print(f"[DEBUG] Episode {ep_idx} metadata: {episodes[-1]}")
        
        # print(f"[DEBUG] Completed loading {len(episodes)} episodes metadata")
        return episodes

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
        penalty_ratio = 0.1
        if is_final_step:
            R = 0.0 if episode_success else -(penalty_ratio * episode_length)
        else:
            current_reward = -1.0
            future_reward_val = 0.0 if episode_success else -(penalty_ratio * episode_length)
            future_rewards = remaining_steps * (-1.0) + future_reward_val
            R = current_reward + future_rewards

        # Normalize
        min_R = -(episode_length - 1) if episode_success else - math.floor((1+penalty_ratio) * episode_length - 1)
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
        Worker sharding -> Episode shuffling -> Frame sequential read -> Buffer shuffle -> Yield
        """
        worker_info = get_worker_info()
        
        # Determine episodes assigned to current worker
        if worker_info is None:
            # Single process mode
            my_episodes = self.episodes_meta
        else:
            # Multi-process mode: simple splitting
            # Note: split by episode, not by frame, to ensure sequential reading
            full_list = self.episodes_meta
            per_worker = int(math.ceil(len(full_list) / float(worker_info.num_workers)))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, len(full_list))
            my_episodes = full_list[start_idx:end_idx]

        # Shuffle episode processing order for randomization
        # In video mode, we read each video sequentially, but video order can be randomized
        rng = np.random.default_rng(self.seed + (worker_info.id if worker_info else 0))
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
                    raw_row = self.lerobot_dataset[global_idx]

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