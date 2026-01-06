import itertools
import traceback
from typing import Optional

import numpy as np
import torch
from PIL import Image

from torch.utils.data import IterableDataset 

class LeRobotValueDataset(IterableDataset):
    """
    Dataset class for loading LeRobot v2.1 format data and converting to conversation-style format.

    For each episode, provides:
    1. Language instruction: Task name from dataset
    2. Observation: Images from observation.images.{camera_name}
    3. Value (R): Computed cumulative reward from current step to end

    Dataset structure:
    - LeRobot v2.1 format with episodes and frames
    - Features: observation.state, action, observation.images.{cam_name}

    Each episode corresponds to one trajectory.
    """

    def __init__(
        self,
        dataset_name: str,
        transform,
        tokenizer,
        dataset_dir: str,  # Path to LeRobot dataset directory or repo_id
        language_instruction: Optional[str] = None,  # Optional default language instruction
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes = None,  # Optional[int]
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        shuffle: bool = True,
        camera_names = None,  # Optional[list], Camera names to load, default: ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        value_tokenizer = None,  # ValueTokenizer instance for encoding R values
    ) -> None:
        """
        Initialize dataset from LeRobot v2.1 format data.
        Assumes all episodes are successful demonstrations.
        """
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.language_instruction = language_instruction or "perform the task"
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.value_tokenizer = value_tokenizer

        # Default camera names (LeRobot standard)
        if camera_names is None:
            camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        self.camera_names = camera_names

        # Import LeRobot here to avoid import errors if not installed
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise ImportError("LeRobot library is required for LeRobotValueDataset. Install with: pip install lerobot")

        # Load LeRobot dataset
        # LeRobot v2.1 supports loading from repo_id or local path
        try:
            self.lerobot_dataset = LeRobotDataset(dataset_dir, version="v2.1")
        except Exception as e:
            # Fallback: try loading without version parameter for backward compatibility
            try:
                self.lerobot_dataset = LeRobotDataset(dataset_dir)
            except Exception as e2:
                raise ValueError(f"Failed to load LeRobot dataset from {dataset_dir}: {e}, {e2}")

        # Get episode information
        self.episodes = self._load_episodes()

        # Create train/val split
        n_episodes = len(self.episodes)
        val_mask = self._get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = self._downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # Filter episodes by train mask
        self.episodes = [ep for i, ep in enumerate(self.episodes) if train_mask[i]]

        # Calculate global normalization range for consistent binning across episodes
        episode_lengths = [ep['length'] for ep in self.episodes]
        if episode_lengths:
            max_episode_length = max(episode_lengths)
            # For successful episodes: global min_R is -(max_episode_length - 1)
            self.global_min_R = -(max_episode_length - 1)
            self.global_max_R = 0.0
        else:
            self.global_min_R = 0.0
            self.global_max_R = 0.0

        # Create indices: each step in each episode is a sample
        self.indices = []
        for episode_idx, episode in enumerate(self.episodes):
            episode_length = episode['length']
            # Each step in the episode is a sample
            for step in range(episode_length):
                self.indices.append((episode_idx, step))

        # Shuffle indices if needed
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(self.indices)

        self.dataset_length = len(self.indices)

    def _calculate_R_value(self, current_step_idx, episode_length, is_final_step=False, episode_success=True):
        """
        Calculate R value and normalize using global R value range for consistent binning.

        This maintains the original R value computation (cumulative reward from current step to end)
        but uses global normalization bounds to ensure consistent binning across episodes of different lengths.

        R value computation:
        - Default reward per step: -1
        - Final step reward: 0 if successful, -(episode_length + 5) if failed
        - R = sum of rewards from current step to end

        Normalization:
        - Use global min_R and max_R calculated during initialization
        - Linear mapping to (-1, 0) range with epsilon to ensure open interval

        Args:
            current_step_idx: Current step index in episode (0-indexed)
            episode_length: Total length of the episode
            is_final_step: Whether this is the final step
            episode_success: Whether the episode was successful

        Returns:
            normalized_R: R value normalized to (-1, 0) range using global bounds
        """
        remaining_steps = episode_length - current_step_idx - 1

        if is_final_step:
            # Final step reward
            if episode_success:
                R = 0.0
            else:
                R = -(episode_length + 5)
        else:
            # Non-final step: current reward (-1) + sum of future rewards
            current_reward = -1.0

            if episode_success:
                # Future: remaining_steps × (-1) + final reward (0)
                future_rewards = remaining_steps * (-1.0) + 0.0
            else:
                # Future: remaining_steps × (-1) + final reward (-(episode_length + 5))
                future_rewards = remaining_steps * (-1.0) + (-(episode_length + 5))

            R = current_reward + future_rewards

        # Normalize R to (-1, 0) range using global normalization bounds
        # Use global min_R and max_R calculated during initialization
        min_R = self.global_min_R
        max_R = self.global_max_R

        # Linear normalization to (-1, 0) - ensure open interval
        if max_R != min_R:
            # Use epsilon to avoid boundary values and ensure open interval (-1, 0)
            eps = 1e-7
            normalized_R = (R - min_R) / (max_R - min_R) * (1.0 - 2*eps) + (-1.0 + eps)
        else:
            normalized_R = -1.0 + 1e-7  # Slightly above -1 for single-value case

        return normalized_R

    def _load_episodes(self):
        """Load episode information from LeRobot dataset."""
        episodes = []

        # LeRobot dataset length gives number of episodes
        for episode_idx in range(len(self.lerobot_dataset)):
            try:
                episode_data = self.lerobot_dataset[episode_idx]

                # Get episode length from observation.state
                if 'observation.state' in episode_data:
                    episode_length = len(episode_data['observation.state'])
                else:
                    # Fallback: try to get length from first available observation
                    first_obs_key = next((k for k in episode_data.keys() if k.startswith('observation.')), None)
                    if first_obs_key:
                        episode_length = len(episode_data[first_obs_key])
                    else:
                        raise ValueError(f"No observation data found in episode {episode_idx}")

                # Get task information - LeRobot v2.1 may have task as string or list
                task = episode_data.get('task', self.language_instruction)
                if isinstance(task, list) and len(task) > 0:
                    instruction = str(task[0])
                elif isinstance(task, str):
                    instruction = task
                else:
                    instruction = self.language_instruction

                episodes.append({
                    'episode_idx': episode_idx,
                    'length': episode_length,
                    'instruction': instruction,
                    'success': True,  # Assume all LeRobot episodes are successful demonstrations
                })

            except Exception as e:
                print(f"Warning: Failed to load episode {episode_idx}: {e}")
                continue

        if not episodes:
            raise ValueError(f"No valid episodes found in LeRobot dataset {self.dataset_dir}")

        return episodes

    def _get_val_mask(self, n_episodes, val_ratio, seed):
        """Create validation mask for episodes."""
        val_mask = np.zeros(n_episodes, dtype=bool)
        if val_ratio <= 0:
            return val_mask

        # Have at least 1 episode for validation, and at least 1 episode for train
        n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
        rng = np.random.default_rng(seed=seed)
        val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
        val_mask[val_idxs] = True
        return val_mask

    def _downsample_mask(self, mask, max_n, seed):
        """Downsample training episodes if needed."""
        train_mask = mask
        if (max_n is not None) and (np.sum(train_mask) > max_n):
            n_train = int(max_n)
            curr_train_idxs = np.nonzero(train_mask)[0]
            rng = np.random.default_rng(seed=seed)
            train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
            train_idxs = curr_train_idxs[train_idxs_idx]
            train_mask = np.zeros_like(train_mask)
            train_mask[train_idxs] = True
        return train_mask

    def _load_frame(self, episode_idx, step):
        """Load a single frame from an episode.

        Args:
            episode_idx: Index in self.episodes list
            step: Step index in the episode
        """
        episode = self.episodes[episode_idx]
        lerobot_episode_idx = episode['episode_idx']

        # Load episode data from LeRobot dataset
        episode_data = self.lerobot_dataset[lerobot_episode_idx]

        if step >= episode['length']:
            raise ValueError(f"Step out of bounds: step={step}, episode_length={episode['length']}")

        # Extract images for each camera
        image_data = {}
        for cam_name in self.camera_names:
            img_key = f"observation.images.{cam_name}"
            if img_key in episode_data:
                # LeRobot v2.1 stores images as numpy arrays
                img_array = episode_data[img_key][step]

                # Handle different image formats
                if isinstance(img_array, np.ndarray):
                    # Check shape: LeRobot typically stores as (C, H, W) or (H, W, C)
                    if len(img_array.shape) == 3:
                        if img_array.shape[0] in [1, 3, 4]:  # Channel-first format (C, H, W)
                            img_array = np.transpose(img_array, (1, 2, 0))  # Convert to (H, W, C)
                        # If already (H, W, C), keep as is
                    elif len(img_array.shape) == 2:  # Grayscale (H, W)
                        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

                    image_data[cam_name] = img_array
            else:
                # Camera not available in this dataset
                print(f"Warning: Camera {cam_name} not found in episode {episode_idx}")

        # Extract state (optional)
        state = None
        if 'observation.state' in episode_data:
            state_data = episode_data['observation.state'][step]
            if isinstance(state_data, np.ndarray):
                state = state_data
            else:
                state = np.array(state_data)

        result = {
            'images': image_data,
            'state': state,
            '_episode_idx': episode_idx,
            '_episode_length': episode['length'],
            '_current_step_in_episode': step,
            '_is_final_step': (step == episode['length'] - 1),
            '_episode_success': episode['success'],  # Always True for LeRobot
        }

        return result

    def batch_transform(self, lerobot_sample):
        """Transform LeRobot sample to conversation format for value prediction."""
        # Extract data
        images = lerobot_sample.get('images', {})
        state = lerobot_sample.get('state', None)

        # Get episode information for R value calculation
        episode_idx = lerobot_sample.get('_episode_idx', None)
        episode_length = lerobot_sample.get('_episode_length', None)
        current_step_in_episode = lerobot_sample.get('_current_step_in_episode', None)
        is_final_step = lerobot_sample.get('_is_final_step', False)
        episode_success = lerobot_sample.get('_episode_success', True)

        # Validate episode info
        if episode_idx is None or episode_length is None or current_step_in_episode is None:
            raise ValueError(
                f"Missing episode info: episode_idx={episode_idx}, "
                f"episode_length={episode_length}, current_step_in_episode={current_step_in_episode}"
            )

        if current_step_in_episode < 0 or current_step_in_episode >= episode_length:
            raise ValueError(
                f"Invalid step index: current_step_in_episode={current_step_in_episode}, "
                f"episode_length={episode_length}"
            )

        # Calculate R value
        R_value = self._calculate_R_value(
            current_step_in_episode,
            episode_length,
            is_final_step,
            episode_success
        )

        # Get language instruction for this episode
        if episode_idx is None or episode_idx >= len(self.episodes):
            raise ValueError(f"Invalid episode_idx: {episode_idx}, total episodes: {len(self.episodes)}")

        episode = self.episodes[episode_idx]
        lang_instruction = episode['instruction']

        # Encode R value using value_tokenizer
        if self.value_tokenizer is not None:
            R_array = np.array([R_value])
            value_str = self.value_tokenizer(R_array)
        else:
            value_str = str(R_value)

        # Convert images to PIL format
        image_list = []
        for cam_name in self.camera_names:
            if cam_name in images:
                img_array = images[cam_name]

                # Ensure uint8 and proper range
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
                    else:
                        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

                # Convert to PIL Image
                if img_array.shape[2] == 3:
                    image_list.append(Image.fromarray(img_array, 'RGB'))
                elif img_array.shape[2] == 4:
                    image_list.append(Image.fromarray(img_array, 'RGBA'))
                else:
                    raise ValueError(f"Unsupported image channel count: {img_array.shape[2]}. "
                                   f"Expected 3 (RGB) or 4 (RGBA) channels, got shape {img_array.shape}")

        # Construct conversation for value prediction
        conversation = []
        conversation.append({
            "from": "human",
            "value": f"You are estimating task progress for robotic manipulation.\n\nGiven a task instruction and a single image, estimate the current progress toward completing the task.\n\nObservation: {'<image>'*len(image_list)}\n\nInstruction: {lang_instruction}",
        })
        conversation.append({
            "from": "gpt",
            "value": value_str,  # Encoded R value
        })

        output_term = {
            "image": image_list,
            "conversations": conversation,
            "value": R_value,  # Keep original R value for reference
            "state": state,  # Keep state for reference if needed
        }

        return output_term

    def __iter__(self):
        """Iterate over dataset with distributed sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Calculate global worker id and worker number
        global_worker_id = self.local_rank * num_workers + worker_id
        num_total_workers = self.world_size * num_workers

        # Create sharded indices
        sharded_indices = itertools.islice(
            range(len(self.indices)),
            global_worker_id,
            None,
            num_total_workers
        )

        for idx in sharded_indices:
            try:
                # Get episode_idx and step from indices
                episode_idx, step = self.indices[idx]

                # Load single frame from episode
                lerobot_sample = self._load_frame(episode_idx, step)

                # Transform to conversation format
                data = self.batch_transform(lerobot_sample)

                # Build Qwen-compatible format
                qwen_data = {
                    "conversations": data['conversations'],
                    "data_path": "",
                    "image": data['image'],  # Already PIL Image list
                }

                yield qwen_data

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing batch at index {idx}: {e}")
                continue

    def __len__(self) -> int:
        """Return dataset length."""
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        """Explicitly unused for IterableDataset."""
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")