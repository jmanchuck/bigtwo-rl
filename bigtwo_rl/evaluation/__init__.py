"""Evaluation and tournament system for Big Two agents."""

from .tournament import Tournament, play_single_game, run_round_robin_tournament
from .evaluator import Evaluator

__all__ = [
    "Tournament",
    "play_single_game", 
    "run_round_robin_tournament",
    "Evaluator",
]