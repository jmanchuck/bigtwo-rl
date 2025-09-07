"""Reward functions for Big Two RL training."""

from .base_reward import BaseReward
from .move_quality_reward_optimized import MoveQualityRewardOptimized
from .sparse_rewards import (
    AggressivePenaltyReward,
    DefaultReward,
    ProgressiveReward,
    RankingReward,
    ScoreMarginReward,
    SparseReward,
)
from .strategic_rewards import (
    ComplexMoveReward,
    StrategicReward,
)
from .zero_sum_reward import ZeroSumReward

__all__ = [
    # Base class
    "BaseReward",
    # Move quality rewards
    "MoveQualityRewardOptimized",
    # Sparse/simple rewards
    "DefaultReward",
    "SparseReward",
    "AggressivePenaltyReward",
    "ProgressiveReward",
    "RankingReward",
    "ScoreMarginReward",
    # Strategic rewards
    "StrategicReward",
    "ComplexMoveReward",
    # Zero-sum reward
    "ZeroSumReward",
]
