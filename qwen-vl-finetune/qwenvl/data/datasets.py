import os
import itertools
import traceback
import json
import random
import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import h5py
import cv2
from PIL import Image

from torch.utils.data import IterableDataset 

# Import from rlds module
from .rlds import make_interleaved_dataset
from .rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from .rlds.utils.data_utils import NormalizationType


class OpenXValueDataset(IterableDataset):
    def __init__(
        self,
        dataset_name,   # 'open_x_embodiment'
        transform,
        # vit_transform,
        tokenizer,
        data_dir_list,  # ['/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/robot/Dataset/OpenVLA_Dataset']
        data_root_dir: Path, # '/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/robot/Dataset/OpenVLA_Dataset'
        data_mix: str, # 'bridge_rt_1'
        resize_resolution: Tuple[int, int], # [256, 256]
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        data_status = None,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        value_tokenizer = None,  # ValueTokenizer instance for encoding R values
        max_episode_length: int = 1000,  # Maximum expected episode length for global normalization
    ) -> None:
        """
        Dataset for value function learning.
        Computes R values (cumulative reward from current step to end) and encodes them using value_tokenizer.
        Each step has reward -1, final successful step has reward 0, final failed step has reward -(episode_length + 5).
        """
        self.dataset_name = dataset_name
        self.data_root_dir, self.data_mix = data_root_dir, data_mix
        self.transform = transform
        # self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.local_rank, self.world_size, self.num_workers = local_rank, world_size, num_workers
        self.value_tokenizer = value_tokenizer
        self.max_episode_length = max_episode_length

        # Calculate global normalization range for consistent binning across episodes
        # Based on actual dataset analysis:
        # - RoboTwin datasets: max episode length ~126, so range [-131, 0]
        # - OpenX datasets: highly variable, use configurable max_episode_length
        # We use the worst case (failed episodes) to ensure all R values fit within the range
        #
        # For better accuracy, consider:
        # 1. Pre-analyzing your specific datasets to get exact max lengths
        # 2. Using the statistics from analyze_episode_lengths.py
        # 3. Setting max_episode_length to match your actual data distribution
        self.global_min_R = -(max_episode_length + 5)
        self.global_max_R = 0.0

        # Configure RLDS Dataset(s), configure the dataset mixture according to OXE_NAMED_MIXTURES
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        # For value learning, we don't need action normalization, but RLDS requires it
        # Use minimal normalization (NORMAL) since we won't use action values
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.NORMAL,  # Minimal normalization, we don't use action
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # Single frame for value prediction
                future_action_window_size=0,                        # No action chunking needed for value learning
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            # For value learning, we need complete episodes, so disable shuffle
            # Set shuffle_buffer_size to 1 to effectively disable shuffling
            shuffle_buffer_size=1 if not train else shuffle_buffer_size,  # Disable shuffle for value learning
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            if "frame_transform_kwargs" not in rlds_config:
                rlds_config["frame_transform_kwargs"] = {}
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs": dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )})
        # fmt: on

        # Initialize RLDS Dataset, we've already set the rlds_config, therefore no need to pay attention to
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

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
    
    def batch_transform(self, rlds_batch):
        """Transform RLDS batch to conversation format for value prediction.
        
        Args:
            rlds_batch: Dict with keys:
                - "observation": dict with "image_primary" and "timestep"
                - "task": dict with "language_instruction"
                - "_episode_length": episode length (added by __iter__)
                - "_current_step": current step in episode (added by __iter__)
                - "_is_final_step": whether this is final step (added by __iter__)
        """
        # Extract required fields - no fallback, raise error if missing
        if not isinstance(rlds_batch, dict):
            raise ValueError(f"rlds_batch must be dict, got {type(rlds_batch)}")
        
        observation = rlds_batch.get("observation")
        if observation is None:
            raise ValueError("rlds_batch missing 'observation' key")
        
        task = rlds_batch.get("task")
        if task is None:
            raise ValueError("rlds_batch missing 'task' key")
        
        # Extract image_primary - required field
        # After chunking with window_size=1, image_primary shape is [1, H, W, C]
        # We need to index [0] to get the actual image [H, W, C]
        img_array = observation.get("image_primary")
        if img_array is None:
            raise ValueError("observation missing 'image_primary' key")
        
        # Extract single image from chunked array
        if isinstance(img_array, np.ndarray):
            if img_array.ndim == 4:
                # Shape [1, H, W, C] - chunked format
                if img_array.shape[0] != 1:
                    raise ValueError(f"With window_size=1, image_primary[0] should have shape [1, H, W, C], got {img_array.shape}")
                img_array = img_array[0]  # Extract [H, W, C]
            elif img_array.ndim == 3:
                # Already [H, W, C] - should not happen with chunking, but handle it
                pass
            else:
                raise ValueError(f"image_primary has unexpected shape: {img_array.shape}")
        else:
            # Handle list/tuple case
            if isinstance(img_array, (list, tuple)):
                if len(img_array) != 1:
                    raise ValueError(f"With window_size=1, image_primary should have length 1, got {len(img_array)}")
                img_array = img_array[0]
        
        img_list = [img_array]  # Convert to list for consistency
        
        # Extract language_instruction - required field
        lang_instruction_raw = task.get("language_instruction")
        if lang_instruction_raw is None:
            raise ValueError("task missing 'language_instruction' key")
        
        # Decode language instruction - deterministic method
        if isinstance(lang_instruction_raw, bytes):
            lang = lang_instruction_raw.decode('utf-8').strip()
        elif isinstance(lang_instruction_raw, str):
            lang = lang_instruction_raw.strip()
        else:
            raise ValueError(f"language_instruction must be bytes or str, got {type(lang_instruction_raw)}")
        
        # Get episode information - required fields added by __iter__
        episode_length = rlds_batch.get('_episode_length')
        if episode_length is None:
            raise ValueError("rlds_batch missing '_episode_length' key")

        current_step = rlds_batch.get('_current_step')
        if current_step is None:
            raise ValueError("rlds_batch missing '_current_step' key")

        is_final_step = rlds_batch.get('_is_final_step', False)
        episode_success = rlds_batch.get('_episode_success', True)  # Default to True for backward compatibility

        # Calculate R value
        R_value = self._calculate_R_value(current_step, episode_length, is_final_step, episode_success)
        
        # Encode R value using value_tokenizer
        if self.value_tokenizer is None:
            raise ValueError("value_tokenizer is required but not set")
        
        R_array = np.array([R_value])
        value_str = self.value_tokenizer(R_array)
        
        # Convert images to PIL format
        image_list = []
        if not isinstance(img_list, (list, tuple)):
            img_list = [img_list]

        for img in img_list:
            if isinstance(img, np.ndarray):
                image_list.append(Image.fromarray(img))
            elif isinstance(img, Image.Image):
                image_list.append(img)
            else:
                raise ValueError(f"Image must be numpy array or PIL Image, got {type(img)}")
        
        # Construct Qwen-compatible conversation format for value prediction
        conversation = []
        conversation.append({
            "from": "human",
            "value": f"You are estimating task progress for robotic manipulation.\n\nGiven a task instruction and a single image, estimate the current progress toward completing the task.\n\nObservation: {'<image>'*len(image_list)}\n\nInstruction: {lang}",
        })
        conversation.append({
            "from": "gpt",
            "value": value_str,  # Encoded R value
        })
        
        output_term = {
            "image": image_list,
            "conversations": conversation,
            "value": R_value,  # Keep original R value for reference
        }

        return output_term

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config) # Interleaved dataset, epoch length, dataset statistics
        # Important: dataset = {"observation": {}, "task": {}, "action": [], "dataset_name": []}
        # Note: action is included by RLDS but we don't use it for value learning

    def __iter__(self):
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

        # Create raw iterator
        iterator = self.dataset.as_numpy_iterator()
        # Set up sharded iterator
        sharded_iterator = itertools.islice(iterator, global_worker_id, None, num_total_workers)

        # Track episode information for R value calculation
        # For RLDS data with window_size=1, timestep is a scalar in observation["timestep"]
        # timestep == 0 indicates the start of a new episode
        # We need to buffer frames to get accurate episode length before calculating R values
        current_episode_frames = []  # Store frames of current episode
        episode_has_success_reward = False  # Track if episode has any reward=1.0

        for rlds_batch in sharded_iterator:
            # Validate batch structure - no fallback
            if not isinstance(rlds_batch, dict):
                raise ValueError(f"rlds_batch must be dict, got {type(rlds_batch)}")

            observation = rlds_batch.get("observation")
            if observation is None:
                raise ValueError("rlds_batch missing 'observation' key")

            # Check for success reward in current step
            current_reward = rlds_batch.get("reward", 0.0)
            if isinstance(current_reward, np.ndarray):
                current_reward = float(current_reward.item()) if current_reward.size == 1 else 0.0
            elif hasattr(current_reward, 'numpy'):
                current_reward = float(current_reward.numpy().item()) if current_reward.numpy().size == 1 else 0.0
            else:
                current_reward = float(current_reward)

            # Update episode success status
            if current_reward >= 1.0:  # reward=1.0 indicates success
                episode_has_success_reward = True

            # Extract timestep - deterministic method
            # After chunking with window_size=1, timestep shape is [1] (1D array)
            # We need to extract the scalar value
            timestep_data = observation.get("timestep")
            if timestep_data is None:
                raise ValueError("observation missing 'timestep' key")

            # Convert to int - chunking with window_size=1 gives shape [1]
            if isinstance(timestep_data, np.ndarray):
                if timestep_data.ndim == 1:
                    # Shape [1] - chunked format
                    if timestep_data.shape[0] != 1:
                        raise ValueError(f"With window_size=1, timestep should have shape [1], got {timestep_data.shape}")
                    current_timestep = int(timestep_data[0])
                elif timestep_data.ndim == 0:
                    # Scalar (should not happen with chunking, but handle it)
                    current_timestep = int(timestep_data)
                else:
                    raise ValueError(f"timestep has unexpected shape: {timestep_data.shape}")
            elif hasattr(timestep_data, 'numpy'):
                timestep_np = timestep_data.numpy()
                if timestep_np.ndim == 1:
                    if timestep_np.shape[0] != 1:
                        raise ValueError(f"With window_size=1, timestep should have shape [1], got {timestep_np.shape}")
                    current_timestep = int(timestep_np[0])
                elif timestep_np.ndim == 0:
                    current_timestep = int(timestep_np)
                else:
                    raise ValueError(f"timestep has unexpected shape: {timestep_np.shape}")
            else:
                raise ValueError(f"timestep must be numpy array or tensor, got {type(timestep_data)}")

            # Check if this is the last step of episode
            is_last = rlds_batch.get("is_last", False) or rlds_batch.get("is_terminal", False)
            if isinstance(is_last, np.ndarray):
                is_last = bool(is_last.item()) if is_last.size == 1 else False
            elif hasattr(is_last, 'numpy'):
                is_last = bool(is_last.numpy().item()) if is_last.numpy().size == 1 else False
            else:
                is_last = bool(is_last)

            # Detect new episode: timestep == 0
            if current_timestep == 0:
                # Process previous episode if exists
                if current_episode_frames:
                    episode_length = len(current_episode_frames)
                    # Determine episode success: has any reward=1.0 throughout the episode
                    episode_success = episode_has_success_reward

                    for frame_idx, frame in enumerate(current_episode_frames):
                        frame['_episode_length'] = episode_length
                        frame['_current_step'] = frame_idx
                        frame['_is_final_step'] = (frame_idx == episode_length - 1)
                        frame['_episode_success'] = episode_success

                        # Transform and yield
                        data = self.batch_transform(frame)
                        qwen_data = {
                            "conversations": data['conversations'],
                            "data_path": "",
                            "image": data['image'],  # Already PIL Image list from batch_transform
                        }
                        yield qwen_data

                # Start new episode
                current_episode_frames = []
                episode_has_success_reward = False  # Reset for new episode

            # Add frame to current episode
            current_episode_frames.append(rlds_batch)

        # Process last episode
        if current_episode_frames:
            episode_length = len(current_episode_frames)
            # Determine episode success: has any reward=1.0 throughout the episode
            episode_success = episode_has_success_reward

            for frame_idx, frame in enumerate(current_episode_frames):
                frame['_episode_length'] = episode_length
                frame['_current_step'] = frame_idx
                frame['_is_final_step'] = (frame_idx == episode_length - 1)
                frame['_episode_success'] = episode_success

                # Transform and yield
                data = self.batch_transform(frame)
                qwen_data = {
                    "conversations": data['conversations'],
                    "data_path": "",
                    "image": data['image'],  # Already PIL Image list from batch_transform
                }
                yield qwen_data

    def __len__(self) -> int:
        # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
        return self.dataset_length 

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class RoboTwinValueDataset(IterableDataset):
    """
    Dataset class for loading hdf5 format data and converting to conversation-style format.
    Similar to OpenXValueDataset but uses hdf5 files instead of RLDS.
    Computes R values (cumulative reward from current step to end) for value function learning.

    RoboTwin contains expert demonstration data, so all episodes are considered successful.

    For each episode, provides:
    1. Language instruction: Loaded from JSON files (episode{i}.json), only from 'seen' list
    2. Observation: Images from hdf5 files (observation/{camera_name}/rgb)
    3. Value (R): Computed cumulative reward from current step to end (each step -1, final step 0)

    Dataset structure:
    - dataset_dir/data/episode{i}.hdf5: Contains observation images (JPEG encoded)
    - dataset_dir/instructions/episode{i}.json: Contains language instructions with 'seen' list

    Each hdf5 file corresponds to one episode.
    """

    def __init__(
        self,
        dataset_name: str,
        transform,
        tokenizer,
        dataset_dir: str,  # Path to dataset directory (e.g., beat_hammer/), contains 'data' and 'instruction' subdirectories
        language_instruction: Optional[str] = None,  # Optional default language instruction if JSON files not found
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes = None,  # Optional[int]
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        shuffle: bool = True,
        camera_names = None,  # Optional[list], Camera names to load, default: ['head_camera']
        value_tokenizer = None,  # ValueTokenizer instance for encoding R values
    ) -> None:
        """
        Initialize dataset from hdf5 format data.
        Each step has reward -1, final successful step has reward 0.
        Only processes hdf5 files from the 'data' directory.
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

        # Default camera names to load
        if camera_names is None:
            camera_names = ['head_camera']
        self.camera_names = camera_names

        # Auto-detect subdirectories
        data_dir = os.path.join(dataset_dir, 'data')
        instruction_dir = os.path.join(dataset_dir, 'instructions')

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        # Load hdf5 data metadata (pass seed for instruction selection)
        self.episodes = self._load_episodes(data_dir, instruction_dir, camera_names, seed)

        # Create train/val split
        n_episodes = len(self.episodes)
        val_mask = self._get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = self._downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # Filter episodes by train mask
        self.episodes = [ep for i, ep in enumerate(self.episodes) if train_mask[i]]

        # Calculate global normalization range for consistent binning across episodes
        # For successful episodes: R ranges from -(episode_length - 1) to 0
        # For failed episodes: R ranges from -(episode_length + 5) to 0
        # Use the worst case (failed episodes) to ensure all R values fit within the range
        episode_lengths = [ep['length'] for ep in self.episodes]
        if episode_lengths:
            max_episode_length = max(episode_lengths)
            # For failed episodes: global min_R is -(max_episode_length + 5)
            self.global_min_R = -(max_episode_length + 5)
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

    def _load_episodes(self, data_dir, instruction_dir, camera_names, seed):
        """Load episode information from hdf5 files and JSON instructions.

        Each hdf5 file corresponds to one episode.
        """
        # Find all hdf5 files and sort by episode number
        hdf5_files = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.hdf5'):
                match = re.search(r'episode(\d+)', filename)
                if match:
                    episode_num = int(match.group(1))
                    hdf5_files.append((episode_num, os.path.join(data_dir, filename)))

        hdf5_files.sort(key=lambda x: x[0])

        if len(hdf5_files) == 0:
            raise FileNotFoundError(f"No hdf5 files found in {data_dir}")

        episodes = []
        rng = random.Random(seed)  # Deterministic instruction selection

        for episode_num, hdf5_file in hdf5_files:
            # Load episode length from hdf5 (using preprocess_aloha.py method)
            with h5py.File(hdf5_file, "r") as f:
                # Use first camera to get length (as in preprocess_aloha.py line 37)
                cam_name = camera_names[0]
                rgb_data = f[f"observation/{cam_name}/rgb"]
                episode_length = len(rgb_data)

                if episode_length == 0:
                    raise ValueError(f"Episode {episode_num} has zero length in {hdf5_file}")

            # Load instruction from JSON (only from 'seen' list)
            if not os.path.exists(instruction_dir):
                raise FileNotFoundError(f"Instruction directory not found: {instruction_dir}")

            json_path = os.path.join(instruction_dir, f"episode{episode_num}.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Instruction file not found: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            seen_list = data.get("seen", [])
            if not seen_list:
                raise ValueError(f"No 'seen' instructions found in {json_path}")

            instruction = rng.choice(seen_list)

            episodes.append({
                'episode_num': episode_num,
                'hdf5_file': hdf5_file,
                'length': episode_length,
                'instruction': instruction
            })

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


    def _calculate_R_value(self, current_step_idx, episode_length, is_final_step=False, episode_success=True):
        """
        Calculate R value (cumulative reward from current step to end) and normalize to (-1, 0).

        According to specification:
        - Default reward per step: -1
        - Final step reward: 0 if successful, -(episode_length + 5) if failed
        - R = sum of rewards from current step to end

        Example for 10-step episode:
        - Step 0 (successful): R = -1 + (-1 × 8) + 0 = -9, normalized to -1.0
        - Step 1 (successful): R = -1 + (-1 × 7) + 0 = -8, normalized to -0.778
        - ...
        - Step 9 (successful): R = 0, normalized to 0.0

        Args:
            current_step_idx: Current step index in episode (0-indexed)
            episode_length: Total length of the episode
            is_final_step: Whether this is the final step
            episode_success: Whether the episode was successful

        Returns:
            normalized_R: R value normalized to (-1, 0) range
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

        # Normalize R to (-1, 0) range using global normalization for consistent binning
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

    def _load_frame(self, episode_idx, step):
        """Load a single frame from an episode.
        
        Args:
            episode_idx: Index in self.episodes list
            step: Step index in the episode
        """
        episode = self.episodes[episode_idx]
        hdf5_file = episode['hdf5_file']
        episode_length = episode['length']
        
        if step >= episode_length:
            raise ValueError(f"Step out of bounds: step={step}, episode_length={episode_length}")
        
        with h5py.File(hdf5_file, "r") as f:
            # Load image using preprocess_aloha.py method (observation/{cam_name}/rgb)
            cam_name = self.camera_names[0]
            rgb_data = f[f"observation/{cam_name}/rgb"]
            
            # Decode JPEG image
            img_bytes = rgb_data[step]
            # Handle hdf5 string/bytes type
            if isinstance(img_bytes, np.bytes_):
                img_bytes = bytes(img_bytes)
            elif not isinstance(img_bytes, bytes):
                img_bytes = bytes(img_bytes)
            
            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_array is None:
                raise ValueError(f"Failed to decode image at step {step} in {hdf5_file}")
            
            # Convert BGR to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        result = {
            'head_camera': img_array,
            '_episode_idx': episode_idx,
            '_episode_length': episode_length,
            '_current_step_in_episode': step,
            '_is_final_step': (step == episode_length - 1)
        }
        
        return result
    
    def batch_transform(self, hdf5_sample):
        """Transform hdf5 sample to conversation format for value prediction."""
        # Extract data (no action needed for value learning)
        head_camera = hdf5_sample.get('head_camera', None)
        state = hdf5_sample.get('state', None)
        
        # Get episode information for R value calculation
        # Episode info should always be available after fix in _sample_sequence
        episode_idx = hdf5_sample.get('_episode_idx', None)
        episode_length = hdf5_sample.get('_episode_length', None)
        current_step_in_episode = hdf5_sample.get('_current_step_in_episode', None)
        is_final_step = hdf5_sample.get('_is_final_step', False)
        
        # Validate episode info - no fallback, raise error if invalid
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
        
        # Calculate R value using episode info
        R_value = self._calculate_R_value(
            current_step_in_episode, 
            episode_length, 
            is_final_step
        )
        
        # Get language instruction for this episode
        if episode_idx is None or episode_idx >= len(self.episodes):
            raise ValueError(f"Invalid episode_idx: {episode_idx}, total episodes: {len(self.episodes)}")
        
        episode = self.episodes[episode_idx]
        lang_instruction = episode['instruction']  # Always loaded from JSON 'seen' list
        
        # Encode R value using value_tokenizer
        # REQUIRE: value_tokenizer must be provided for value function training
        if self.value_tokenizer is None:
            raise ValueError("value_tokenizer is required for value function training. "
                           "Please ensure value_tokenizer is passed to the dataset constructor.")

        # value_tokenizer expects numpy array
        R_array = np.array([R_value])
        value_str = self.value_tokenizer(R_array)
        
        # Convert image to PIL format (single frame, shape: H, W, C)
        image_list = []
        if head_camera is not None:
            # head_camera is a single frame (H, W, C) from _load_frame
            img_array = head_camera
            
            # Handle channel-first format (C, H, W) -> (H, W, C)
            if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3, 4]:
                img_array = np.moveaxis(img_array, 0, -1)
            
            # Handle grayscale (H, W) -> (H, W, 1) -> (H, W, 3)
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            if len(img_array.shape) == 3 and img_array.shape[2] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            
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
    
    def _pil_img2rgb(self, img):
        """Convert PIL image to RGB format."""
        if img.mode == 'RGB':
            return img
        elif img.mode == 'RGBA':
            # Create white background
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            return rgb_img
        elif img.mode == 'L':
            return img.convert('RGB')
        elif img.mode == 'P':
            return img.convert('RGB')
        else:
            return img.convert('RGB')
    
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
                hdf5_sample = self._load_frame(episode_idx, step)
                
                # Transform to conversation format
                data = self.batch_transform(hdf5_sample)
                
                # Build Qwen-compatible format
                # Image is already processed as PIL Image list in batch_transform
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


class OpenPiValueDataset(IterableDataset):
    """
    Dataset class for loading OpenPi format data and converting to conversation-style format.
    Similar to RoboTwinValueDataset but uses JSON format data from OpenPi.

    For each episode, provides:
    1. Language instruction: Loaded from 'prompt' field in trajectory
    2. Observation: Images from trajectory['observation']['images'] with cameras: cam_high, cam_left_wrist, cam_right_wrist
    3. Value (R): Computed cumulative reward from current step to end (each step reward from data, final step based on success)

    Dataset structure (per data/format.md):
    - dataset_dir/results: List of results, each containing task_id, result(success/fail), trajectory
    - trajectory: Contains observation(images, state), prompt, action, reward(0/1), timestep

    Each trajectory corresponds to one episode.
    """

    def __init__(
        self,
        dataset_name: str,
        transform,
        tokenizer,
        dataset_dir: str,  # Path to dataset directory containing results
        language_instruction: Optional[str] = None,  # Optional default language instruction if prompt not found
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
        Initialize dataset from OpenPi format data.
        Each step has reward -1, final successful step has reward 0, final failed step has reward -(episode_length + 5).
        Processes both successful and failed episodes for value learning.
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

        # Default camera names to load (as per format.md)
        if camera_names is None:
            camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        self.camera_names = camera_names

        # Load dataset metadata
        self.episodes = self._load_episodes(dataset_dir, seed)

        # Create train/val split
        n_episodes = len(self.episodes)
        val_mask = self._get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = self._downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # Filter episodes by train mask
        self.episodes = [ep for i, ep in enumerate(self.episodes) if train_mask[i]]

        # Calculate global normalization range for consistent binning across episodes
        # For successful episodes: R ranges from -(episode_length - 1) to 0
        # For failed episodes: R ranges from -(episode_length + 5) to 0
        # OpenPi may have both successful and failed episodes
        episode_lengths = [ep['length'] for ep in self.episodes]
        success_flags = [ep['success'] for ep in self.episodes]
        if episode_lengths:
            max_episode_length = max(episode_lengths)
            # For episodes with both success and failure, use the worst case (failed episodes)
            # Failed episodes have larger negative R values: -(episode_length + 5)
            self.global_min_R = -(max_episode_length + 5)
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

    def _load_episodes(self, dataset_dir, seed):
        """Load episode information from OpenPi pickle file.

        Processes all episodes (both successful and failed) from the pickle file.
        """
        import pickle

        # Find pickle file
        pickle_files = []
        for filename in os.listdir(dataset_dir):
            if filename.endswith('.pkl'):
                pickle_files.append(os.path.join(dataset_dir, filename))

        if len(pickle_files) == 0:
            raise FileNotFoundError(f"No pickle files found in {dataset_dir}")

        # For now, use the first pickle file (or the specific one if provided)
        pickle_file = pickle_files[0]
        if len(pickle_files) > 1:
            # If multiple files, prefer the one matching the dataset name or use the first one
            target_name = os.path.basename(dataset_dir)
            for pf in pickle_files:
                if target_name in os.path.basename(pf):
                    pickle_file = pf
                    break

        print(f"Loading OpenPi dataset from: {pickle_file}")

        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # Extract results
        results = data.get('results', [])
        if not results:
            raise ValueError(f"No results found in {pickle_file}")

        episodes = []
        episode_counter = 0

        for result in results:
            # Process all episodes (both successful and failed)
            trajectory = result.get('trajectory', [])
            if not trajectory:
                continue

            episode_length = len(trajectory)
            if episode_length == 0:
                continue

            # Extract task_id for identification
            task_id = result.get('task_id', f'episode_{episode_counter}')

            # Get episode success status
            episode_success = result.get('result', False)

            # Use the first trajectory item to get the prompt from observation
            prompt = self.language_instruction
            if trajectory:
                observation = trajectory[0].get('observation', {})
                prompt = observation.get('prompt', self.language_instruction)
                if prompt is None:
                    prompt = self.language_instruction

            episodes.append({
                'episode_num': episode_counter,
                'pickle_file': pickle_file,
                'task_id': task_id,
                'length': episode_length,
                'instruction': prompt,
                'trajectory': trajectory,  # Store full trajectory data
                'success': episode_success,  # Store episode success status
            })

            episode_counter += 1

        if len(episodes) == 0:
            raise ValueError(f"No valid episodes found in {pickle_file}")

        print(f"Loaded {len(episodes)} successful episodes from {len(results)} total results")
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

    def _calculate_R_value(self, current_step_idx, episode_length, is_final_step=False, episode_success=True):
        """
        Calculate R value (cumulative reward from current step to end) and normalize to (-1, 0).

        Each step has reward -1, final successful step has reward 0, final failed step has reward -(episode_length + 5).
        For OpenPi, we support both successful and failed episodes.
        Example for 200-step episode:
        - Step 0 (first): R = -199 (199 remaining steps × -1 + 0), normalized to -1.0
        - Step 1: R = -198, normalized to -0.995
        - ...
        - Step 199 (last, successful): R = 0, normalized to 0.0
        - Step 199 (last, failed): R = -(200 + 5) = -205, normalized accordingly

        Args:
            current_step_idx: Current step index in episode (0-indexed)
            episode_length: Total length of the episode
            is_final_step: Whether this is the final step
            episode_success: Whether the episode was successful (default: True for backward compatibility)

        Returns:
            normalized_R: R value normalized to (-1, 0) range
        """
        # Calculate remaining steps (excluding current step)
        remaining_steps = episode_length - current_step_idx - 1

        # R = sum of rewards from current step to end
        if is_final_step:
            # Final step reward depends on episode success
            if episode_success:
                R = 0.0  # Successful completion
            else:
                R = -(episode_length + 5)  # Failed completion penalty
        else:
            # Non-final steps: remaining steps × (-1) + final step reward
            if episode_success:
                R = -remaining_steps  # Final reward = 0
            else:
                R = -remaining_steps + (-(episode_length + 5))  # Final reward = -(episode_length + 5)

        # Normalize R to (-1, 0) range using global normalization for consistent binning
        # Use global min_R and max_R calculated during initialization
        min_R = self.global_min_R
        max_R = self.global_max_R

        # Normalize to (-1, 0) using linear scaling - ensure open interval
        # Use epsilon to avoid boundary values and ensure open interval (-1, 0)
        if max_R != min_R:
            eps = 1e-7
            normalized_R = (R - min_R) / (max_R - min_R) * (1.0 - 2*eps) + (-1.0 + eps)
        else:
            normalized_R = -1.0 + 1e-7  # Slightly above -1 for single-value case

        return normalized_R

    def _load_frame(self, episode_idx, step):
        """Load a single frame from an episode trajectory.

        Args:
            episode_idx: Index in self.episodes list
            step: Step index in the episode
        """
        episode = self.episodes[episode_idx]
        trajectory = episode['trajectory']
        episode_length = episode['length']
        episode_success = episode['success']  # Get success status from episode

        if step >= episode_length:
            raise ValueError(f"Step out of bounds: step={step}, episode_length={episode_length}")

        # Get the trajectory step data
        step_data = trajectory[step]

        # Extract images from observation
        observation = step_data.get('observation', {})
        images = observation.get('images', {})

        # Load images for each camera
        image_data = {}
        for cam_name in self.camera_names:
            if cam_name in images:
                img_array = images[cam_name]  # Already numpy array with shape (3, 240, 320)

                # Convert from (C, H, W) to (H, W, C) for PIL
                if img_array.shape[0] == 3:  # RGB channels first
                    img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, C)
                else:
                    # Handle other channel configurations if needed
                    img_array = np.transpose(img_array, (1, 2, 0))

                image_data[cam_name] = img_array

        # Extract state if available
        state = observation.get('state')

        # Extract other metadata
        reward = step_data.get('reward', False)
        # Convert boolean reward to numeric (True=1, False=-1 for unsuccessful steps)
        reward_numeric = 1.0 if reward else -1.0

        timestep = step_data.get('timestep', step)

        result = {
            'images': image_data,
            'state': state,
            'reward': reward_numeric,  # Converted to numeric
            'timestep': timestep,
            '_episode_idx': episode_idx,
            '_episode_length': episode_length,
            '_current_step_in_episode': step,
            '_is_final_step': (step == episode_length - 1),
            '_episode_success': episode_success,  # Use actual episode success status
        }

        return result

    def batch_transform(self, openpi_sample):
        """Transform OpenPi sample to conversation format for value prediction."""
        # Extract data
        images = openpi_sample.get('images', {})
        state = openpi_sample.get('state', None)

        # Get episode information for R value calculation
        episode_idx = openpi_sample.get('_episode_idx', None)
        episode_length = openpi_sample.get('_episode_length', None)
        current_step_in_episode = openpi_sample.get('_current_step_in_episode', None)
        is_final_step = openpi_sample.get('_is_final_step', False)
        episode_success = openpi_sample.get('_episode_success', False)

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
                openpi_sample = self._load_frame(episode_idx, step)

                # Transform to conversation format
                data = self.batch_transform(openpi_sample)

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