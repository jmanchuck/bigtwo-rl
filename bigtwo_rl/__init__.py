"""Big Two RL Agent Library

A reinforcement learning library for training and evaluating AI agents that play Big Two.

Main components:
- bigtwo_rl.core: Game environment and RL wrapper
- bigtwo_rl.agents: Various agent implementations
- bigtwo_rl.training: Training infrastructure with configurable rewards/hyperparams
- bigtwo_rl.evaluation: Tournament system and evaluation tools
"""

__version__ = "0.1.0"

# Main API exports
from .core import BigTwoRLWrapper, ToyBigTwoFullRules
from .agents import BaseAgent, RandomAgent, GreedyAgent, PPOAgent

__all__ = [
    "BigTwoRLWrapper",
    "ToyBigTwoFullRules",
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "PPOAgent",
]
