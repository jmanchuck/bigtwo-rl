"""Training infrastructure for Big Two agents."""

from .trainer import Trainer
from .rewards import BaseReward, get_reward_function, list_reward_functions
from .opponent_pool import OpponentPool
from .hyperparams import get_config, list_configs

__all__ = [
    "Trainer",
    "BaseReward",
    "get_reward_function",
    "list_reward_functions",
    "get_config",
    "list_configs",
    "OpponentPool",
]
