"""Evaluation and tournament system for Big Two agents."""

from .tournament import Tournament, SeriesEvaluator
from .evaluator import Evaluator

__all__ = [
    "Tournament",
    "SeriesEvaluator", 
    "Evaluator",
]
