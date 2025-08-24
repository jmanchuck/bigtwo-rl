"""Agents package for Big Two RL training."""

from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .greedy_agent import GreedyAgent
from .ppo_agent import PPOAgent
from .human_agent import HumanAgent
import os
from typing import Optional


def load_ppo_agent(model_path: str, name: Optional[str] = None) -> PPOAgent:
    """Convenience function to load a PPO agent with automatic naming.

    Args:
        model_path: Path to the trained model (e.g., "./models/my_model/best_model")
        name: Optional name for the agent (defaults to model directory name)

    Returns:
        PPOAgent instance

    Example:
        agent = load_ppo_agent("./models/my_model/best_model")
        # or with custom name:
        agent = load_ppo_agent("./models/my_model/best_model", "MyAgent")
    """
    if name is None:
        # Extract name from model path - use parent directory name
        name = (
            os.path.basename(os.path.dirname(model_path))
            if "/" in model_path
            else "PPO"
        )

    return PPOAgent(model_path=model_path, name=name)


__all__ = [
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "PPOAgent",
    "HumanAgent",
    "load_ppo_agent",
]
