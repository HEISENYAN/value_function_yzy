"""
Script to convert RoboTwin hdf5 data directly to LeRobot dataset v2.1 format without intermediate files.
Combines data processing and format conversion into a single streamlined pipeline.
"""

import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

import numpy as np
import torch
import tqdm
import tyro
import json
import fnmatch
import cv2
import argparse

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 224, 224),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_aloha_episode_data(
    ep_path: Path,
    mode: str = "image",
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Load and process raw Aloha episode data directly from original HDF5 format."""

    # Load raw data from original HDF5 format 
    if not os.path.isfile(ep_path):
        raise FileNotFoundError(f"Dataset does not exist at {ep_path}")

    with h5py.File(ep_path, "r") as root:
        left_gripper_all = root["/joint_action/left_gripper"][()]
        left_arm_all = root["/joint_action/left_arm"][()]
        right_gripper_all = root["/joint_action/right_gripper"][()]
        right_arm_all = root["/joint_action/right_arm"][()]

        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    # Process data in memory (adapted from process_data.py data_transform logic)
    qpos = []
    actions = []
    cam_high = []
    cam_right_wrist = []
    cam_left_wrist = []

    for j in range(left_gripper_all.shape[0]):
        left_gripper, left_arm, right_gripper, right_arm = (
            left_gripper_all[j],
            left_arm_all[j],
            right_gripper_all[j],
            right_arm_all[j],
        )

        # Create state vector (joint angles)
        state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])
        state = state.astype(np.float32)

        # Process observations (all frames except last)
        if j != left_gripper_all.shape[0] - 1:
            qpos.append(state)

            # Process images for each camera
            camera_mappings = {
                "head_camera": "cam_high",
                "right_camera": "cam_right_wrist",
                "left_camera": "cam_left_wrist"
            }

            for orig_cam, lerobot_cam in camera_mappings.items():
                camera_bits = image_dict[orig_cam][j]
                camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                # Resize to 224x224 for storage and future use
                camera_resized = cv2.resize(camera_img, (224, 224))

                if lerobot_cam == "cam_high":
                    cam_high.append(camera_resized)
                elif lerobot_cam == "cam_right_wrist":
                    cam_right_wrist.append(camera_resized)
                elif lerobot_cam == "cam_left_wrist":
                    cam_left_wrist.append(camera_resized)

        # Process actions (all frames except first)
        if j != 0:
            action = state
            actions.append(action)

    # Convert to torch tensors
    state_tensor = torch.from_numpy(np.array(qpos))
    action_tensor = torch.from_numpy(np.array(actions))

    # Organize images by camera - always use decoded image arrays
    # LeRobot will handle compression/encoding internally based on mode and use_videos settings
    imgs_per_cam = {
        "cam_high": np.array(cam_high),
        "cam_right_wrist": np.array(cam_right_wrist),
        "cam_left_wrist": np.array(cam_left_wrist),
    }

    # For now, we don't handle velocity and effort in the original format
    velocity = None
    effort = None

    return imgs_per_cam, state_tensor, action_tensor, velocity, effort


def populate_dataset_from_aloha(
    dataset: LeRobotDataset,
    raw_dir: Path,
    episodes: list[int] | None = None,
    mode: str = "image",
) -> LeRobotDataset:
    """Populate LeRobot dataset directly from raw Aloha data."""

    # Find all HDF5 files
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            file_path = os.path.join(root, filename)
            hdf5_files.append(Path(file_path))

    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        # Load processed episode data directly
        imgs_per_cam, state, action, velocity, effort = load_raw_aloha_episode_data(ep_path, mode)
        num_frames = state.shape[0]

        # Load instructions for this episode
        dir_path = os.path.dirname(ep_path)
        json_path = f"{dir_path}/../instructions/episode{ep_idx}.json"

        instructions = ["default_task"]  # fallback
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f_instr:
                    instruction_dict = json.load(f_instr)
                    instructions = instruction_dict.get("seen", ["default_task"])
            except (json.JSONDecodeError, KeyError):
                raise ValueError(f"Failed to load instructions from {json_path}")

        # Add frames to dataset
        for i in range(num_frames):
            frame = {"observation.state": state[i], "action": action[i]}

            # Add images for each camera
            for camera, img_array in imgs_per_cam.items():
                # Always convert BGR to RGB and transpose for LeRobot format (C, H, W)
                # LeRobot will handle video encoding internally if use_videos=True
                img_rgb = cv2.cvtColor(img_array[i], cv2.COLOR_BGR2RGB)
                img_transposed = np.transpose(img_rgb, (2, 0, 1))
                frame[f"observation.images.{camera}"] = img_transposed

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            # Randomly select instruction for this frame
            instruction = np.random.choice(instructions)
            dataset.add_frame(frame, task=instruction)

        dataset.save_episode()

    return dataset


def port_aloha_direct(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert Aloha data directly to LeRobot format without intermediate files."""

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")

    # Find HDF5 files to check data types
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {raw_dir}")

    # Check data types (simplified - assuming no velocity/effort in raw format)
    has_velocity = False
    has_effort = False

    # Create empty dataset
    dataset = create_empty_dataset(
        repo_id,
        robot_type="aloha",
        mode=mode,
        has_effort=has_effort,
        has_velocity=has_velocity,
        dataset_config=dataset_config,
    )

    # Populate dataset directly from raw data
    dataset = populate_dataset_from_aloha(
        dataset,
        raw_dir,
        episodes=episodes,
        mode=mode,
    )

    if push_to_hub:
        dataset.push_to_hub()


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Convert Aloha data directly to LeRobot format.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw Aloha data directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobot dataset repository ID"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs='*',
        help="Specific episode indices to process (optional)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="image",
        help="Dataset mode"
    )

    args = parser.parse_args()

    episodes = args.episodes if args.episodes else None

    port_aloha_direct(
        raw_dir=Path(args.raw_dir),
        repo_id=args.repo_id,
        episodes=episodes,
        push_to_hub=args.push_to_hub,
        mode=args.mode,
    )


if __name__ == "__main__":
    # Support both tyro and argparse interfaces
    try:
        tyro.cli(port_aloha_direct)
    except SystemExit:
        # Fall back to argparse if tyro fails
        main()