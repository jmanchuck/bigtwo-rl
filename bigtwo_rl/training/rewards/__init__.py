"""Reward functions for Big Two RL training."""

from .base_reward import BaseReward
from .move_quality_reward import MoveQualityReward
from .sparse_rewards import (
    DefaultReward,
    SparseReward,
    AggressivePenaltyReward,
    ProgressiveReward,
    RankingReward,
    ScoreMarginReward,
)
from .strategic_rewards import (
    StrategicReward,
    ComplexMoveReward,
)

__all__ = [
    # Base class
    "BaseReward",
    
    # Move quality reward
    "MoveQualityReward",
    
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
]