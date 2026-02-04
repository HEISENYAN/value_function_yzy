# Simplified dataset module - only supports LeRobot format

def create_lerobot_dataset(**kwargs):
    """
    Create a LeRobot value dataset instance.

    Args:
        **kwargs: Arguments to pass to LeRobotValueDataset constructor

    Returns:
        LeRobotValueDataset instance
    """
    from .datasets import LeRobotValueDataset
    return LeRobotValueDataset(**kwargs)
