"""Training infrastructure for Big Two agents."""

from .trainer import Trainer
from .rewards import BaseReward
from .opponent_pool import OpponentPool
from .callbacks import BigTwoMetricsCallback

__all__ = [
    "Trainer",
    "BaseReward",
    "OpponentPool", 
    "BigTwoMetricsCallback",
]
