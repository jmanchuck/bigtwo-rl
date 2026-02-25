"""Training infrastructure for Big Two agents."""

# Import reward functions first to avoid circular imports
from .rewards import DefaultReward, ProgressiveReward, SparseReward, StrategicReward, ZeroSumReward

# Import trainer after rewards
from .trainer import Trainer, quick_train

__all__ = [
    "Trainer",
    "quick_train",
    # Reward functions
    "DefaultReward",
    "SparseReward",
    "ZeroSumReward",
    "ProgressiveReward",
    "StrategicReward",
]
