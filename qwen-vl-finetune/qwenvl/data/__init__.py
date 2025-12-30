import re
import os
from typing import Dict, Any, Optional, Tuple

# Dataset type configurations
DATASET_CONFIGS = {
    "robottwin": {
        "dataset_type": "RoboTwin",
        "class_name": "RoboTwinValueDataset",
        "path_patterns": ["robotwin"],
        "file_extensions": [".hdf5"],
        "requires_rlds": False,
    },
    "openpi": {
        "dataset_type": "OpenPi",
        "class_name": "OpenPiValueDataset",
        "path_patterns": ["openpi"],
        "file_extensions": [".pkl"],
        "requires_rlds": False,
    },
    "oxe": {
        "dataset_type": "OXE",
        "class_name": "OpenXValueDataset",
        "path_patterns": ["oxe", "open_x_embodiment"],
        "file_extensions": [],
        "requires_rlds": True,
    },
}

# Legacy dataset configurations for backward compatibility
data_dict = {
    name: {"dataset_type": config["dataset_type"], "data_path": ""}
    for name, config in DATASET_CONFIGS.items()
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def detect_dataset_type(dataset_path: str) -> str:
    """
    Detect dataset type from path or file contents.

    Args:
        dataset_path: Path to dataset directory or file

    Returns:
        Dataset type string (e.g., 'robottwin', 'openpi', 'oxe')
    """
    # Handle file paths - extract directory
    if os.path.isfile(dataset_path):
        dataset_dir = os.path.dirname(dataset_path)
        filename = os.path.basename(dataset_path)
    else:
        dataset_dir = dataset_path
        filename = None

    # Check path-based detection
    path_lower = dataset_path.lower()
    for dataset_type, config in DATASET_CONFIGS.items():
        if any(pattern.lower() in path_lower for pattern in config["path_patterns"]):
            return dataset_type

    # Check file contents if it's a directory
    if os.path.isdir(dataset_dir):
        for dataset_type, config in DATASET_CONFIGS.items():
            # Check for specific file extensions
            for ext in config["file_extensions"]:
                if any(f.endswith(ext) for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))):
                    return dataset_type

    # Check specific file extension
    if filename:
        for dataset_type, config in DATASET_CONFIGS.items():
            if any(filename.endswith(ext) for ext in config["file_extensions"]):
                return dataset_type

    raise ValueError(f"Cannot detect dataset type for path: {dataset_path}. "
                    f"No matching file extensions found in supported datasets: {list(DATASET_CONFIGS.keys())}")


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset configuration for a given dataset name or path.

    Args:
        dataset_name: Dataset name or path

    Returns:
        Dataset configuration dictionary
    """
    sampling_rate = parse_sampling_rate(dataset_name)
    dataset_name_clean = re.sub(r"%(\d+)$", "", dataset_name)

    # Detect dataset type
    dataset_type = detect_dataset_type(dataset_name_clean)

    # Get base config
    if dataset_type in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_type].copy()
    elif dataset_name_clean in data_dict:
        # Fallback to legacy config
        config = data_dict[dataset_name_clean].copy()
        config.update(DATASET_CONFIGS.get(config["dataset_type"].lower(), {}))
    else:
        raise ValueError(f"Unknown dataset type for: {dataset_name_clean}")

    # Set common fields
    config.update({
        "data_path": dataset_name_clean,
        "sampling_rate": sampling_rate,
    })

    return config


def data_list(dataset_names):
    """Get configuration list for multiple datasets."""
    return [get_dataset_config(name) for name in dataset_names]


# Dataset factory function
def create_value_dataset(dataset_config: Dict[str, Any], **kwargs) -> Any:
    """
    Create a value dataset instance based on configuration.

    Args:
        dataset_config: Dataset configuration from get_dataset_config()
        **kwargs: Additional arguments to pass to dataset constructor

    Returns:
        Dataset instance
    """
    dataset_type = dataset_config["dataset_type"]
    class_name = dataset_config["class_name"]

    # Import the dataset class dynamically
    if class_name == "RoboTwinValueDataset":
        from .datasets import RoboTwinValueDataset
        dataset_class = RoboTwinValueDataset
    elif class_name == "OpenPiValueDataset":
        from .datasets import OpenPiValueDataset
        dataset_class = OpenPiValueDataset
    elif class_name == "OpenXValueDataset":
        from .datasets import OpenXValueDataset
        dataset_class = OpenXValueDataset
    else:
        raise ValueError(f"Unknown dataset class: {class_name}")

    # For RLDS-based datasets (like OpenX), we need special handling
    if dataset_config.get("requires_rlds", False):
        # OpenXValueDataset has a different constructor signature
        # It needs additional parameters that other datasets don't have
        # This will be handled in data_processor.py where we have access to all args
        raise ValueError(f"RLDS-based datasets like {dataset_type} should be created directly in data_processor.py")

    # Create dataset instance
    return dataset_class(**kwargs)


# Legacy functions for backward compatibility
def is_robottwin_path(path):
    """Check if a path is a RoboTwin dataset path."""
    return detect_dataset_type(path) == "robottwin"


def is_openpi_path(path):
    """Check if a path is an OpenPi dataset path."""
    return detect_dataset_type(path) == "openpi"


if __name__ == "__main__":
    dataset_names = ["openpi"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
