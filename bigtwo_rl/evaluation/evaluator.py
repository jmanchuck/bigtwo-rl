"""Evaluator for Big Two models.

This module provides comprehensive evaluation capabilities for models trained
with the 1,365-action space, including performance analysis and comparison.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.greedy_agent import create_basic_greedy_agent, create_smart_greedy_agent
from ..agents.ppo_agent import PPOAgent
from ..agents.random_agent import create_balanced_random_agent
from .tournament import SeriesEvaluator


class Evaluator:
    """Evaluator for Big Two models.

    This evaluator provides comprehensive performance analysis against various baselines
    using the 1,365-action space system.
    """

    def __init__(
        self,
        num_games: int = 100,
        n_processes: int | None = None,
        verbose: bool = True,
    ):
        """Initialize Evaluator.

        Args:
            num_games: Number of games per evaluation
            n_processes: Number of processes for parallel evaluation
            verbose: Whether to print progress information

        """
        self.num_games = num_games
        self.n_processes = n_processes
        self.verbose = verbose
        self.series_evaluator = SeriesEvaluator(verbose=verbose)

        if self.verbose:
            print("🔍 Evaluator initialized")
            print(f"🎮 Games per evaluation: {num_games}")
            print(f"⚡ Parallel processes: {n_processes or 'auto-detect'}")

    def evaluate_model(
        self,
        model_path: str,
        opponents: list[str] | None = None,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a fixed action space model.

        Args:
            model_path: Path to trained PPO model
            opponents: List of opponent types ['random', 'greedy', 'smart_greedy']
            deterministic: Whether to use deterministic actions

        Returns:
            Comprehensive evaluation results

        """
        if self.verbose:
            print(f"🎯 Evaluating model: {model_path}")

        # Validate model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create test agent
        try:
            test_agent = PPOAgent(
                model_path=model_path,
                name="TestAgent",
                deterministic=deterministic,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Create baseline opponents
        opponents = opponents or ["random", "greedy", "smart_greedy"]
        opponent_agents = self._create_opponent_agents(opponents)

        # Run evaluations against different opponent combinations
        results = {}
        total_start_time = time.time()

        # 1. Evaluation against random opponents
        if "random" in opponents:
            random_opponents = [
                create_balanced_random_agent("Random1"),
                create_balanced_random_agent("Random2"),
                create_balanced_random_agent("Random3"),
            ]
            agents = [test_agent] + random_opponents

            if self.verbose:
                print("🎲 Evaluating vs Random opponents...")

            random_results = self.series_evaluator.play_four_player_series(
                agents,
                self.num_games,
                self.n_processes,
            )
            results["vs_random"] = random_results

        # 2. Evaluation against greedy opponents
        if "greedy" in opponents:
            greedy_opponents = [
                create_basic_greedy_agent("Greedy1"),
                create_basic_greedy_agent("Greedy2"),
                create_basic_greedy_agent("Greedy3"),
            ]
            agents = [test_agent] + greedy_opponents

            if self.verbose:
                print("🧠 Evaluating vs Greedy opponents...")

            greedy_results = self.series_evaluator.play_four_player_series(
                agents,
                self.num_games,
                self.n_processes,
            )
            results["vs_greedy"] = greedy_results

        # 3. Evaluation against smart greedy opponents
        if "smart_greedy" in opponents:
            smart_opponents = [
                create_smart_greedy_agent("SmartGreedy1"),
                create_smart_greedy_agent("SmartGreedy2"),
                create_smart_greedy_agent("SmartGreedy3"),
            ]
            agents = [test_agent] + smart_opponents

            if self.verbose:
                print("🎯 Evaluating vs Smart Greedy opponents...")

            smart_results = self.series_evaluator.play_four_player_series(
                agents,
                self.num_games,
                self.n_processes,
            )
            results["vs_smart_greedy"] = smart_results

        # 4. Mixed opponents evaluation
        if len(opponents) > 1:
            mixed_opponents = []
            if "random" in opponents:
                mixed_opponents.append(create_balanced_random_agent("MixedRandom"))
            if "greedy" in opponents:
                mixed_opponents.append(create_basic_greedy_agent("MixedGreedy"))
            if "smart_greedy" in opponents:
                mixed_opponents.append(create_smart_greedy_agent("MixedSmart"))

            # Fill remaining slots with random agents
            while len(mixed_opponents) < 3:
                mixed_opponents.append(create_balanced_random_agent(f"FillRandom{len(mixed_opponents)}"))

            agents = [test_agent] + mixed_opponents[:3]

            if self.verbose:
                print("🔀 Evaluating vs Mixed opponents...")

            mixed_results = self.series_evaluator.play_four_player_series(
                agents,
                self.num_games,
                self.n_processes,
            )
            results["vs_mixed"] = mixed_results

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Aggregate results
        evaluation_summary = self._create_evaluation_summary(
            results,
            model_path,
            test_agent,
            total_duration,
        )

        if self.verbose:
            self._print_evaluation_summary(evaluation_summary)

        return evaluation_summary

    def _create_opponent_agents(self, opponent_types: list[str]) -> dict[str, list]:
        """Create opponent agents for evaluation.

        Args:
            opponent_types: List of opponent types

        Returns:
            Dictionary of opponent agents by type

        """
        opponents = {}

        for opp_type in opponent_types:
            if opp_type == "random":
                opponents[opp_type] = [create_balanced_random_agent(f"Random{i}") for i in range(3)]
            elif opp_type == "greedy":
                opponents[opp_type] = [create_basic_greedy_agent(f"Greedy{i}") for i in range(3)]
            elif opp_type == "smart_greedy":
                opponents[opp_type] = [create_smart_greedy_agent(f"SmartGreedy{i}") for i in range(3)]
            else:
                print(f"⚠️  Unknown opponent type: {opp_type}")

        return opponents

    def _create_evaluation_summary(
        self,
        results: dict[str, Any],
        model_path: str,
        test_agent: PPOAgent,
        total_duration: float,
    ) -> dict[str, Any]:
        """Create comprehensive evaluation summary.

        Args:
            results: Raw evaluation results
            model_path: Path to evaluated model
            test_agent: Test agent instance
            total_duration: Total evaluation time

        Returns:
            Evaluation summary dictionary

        """
        summary = {
            "model_path": model_path,
            "model_info": test_agent.get_model_info(),
            "evaluation_config": {
                "num_games_per_scenario": self.num_games,
                "total_games": len(results) * self.num_games,
                "scenarios_evaluated": list(results.keys()),
                "n_processes": self.n_processes,
                "total_duration_seconds": total_duration,
            },
            "scenario_results": results,
            "overall_performance": {},
        }

        # Calculate overall performance metrics
        total_wins = 0
        total_games = 0
        total_cards_left = 0
        scenario_win_rates = {}

        for scenario, scenario_results in results.items():
            wins = scenario_results["wins"][0]  # Test agent is always index 0
            games = scenario_results["completed_games"]
            avg_cards = scenario_results["avg_scores"][0]

            total_wins += wins
            total_games += games
            total_cards_left += avg_cards * games

            win_rate = wins / games if games > 0 else 0
            scenario_win_rates[scenario] = win_rate

        # Overall metrics
        overall_win_rate = total_wins / total_games if total_games > 0 else 0
        overall_avg_cards = total_cards_left / total_games if total_games > 0 else 0

        summary["overall_performance"] = {
            "total_wins": total_wins,
            "total_games": total_games,
            "overall_win_rate": overall_win_rate,
            "overall_avg_cards_left": overall_avg_cards,
            "scenario_win_rates": scenario_win_rates,
            "performance_grade": self._calculate_performance_grade(scenario_win_rates),
        }

        return summary

    def _calculate_performance_grade(self, scenario_win_rates: dict[str, float]) -> str:
        """Calculate overall performance grade.

        Args:
            scenario_win_rates: Win rates by scenario

        Returns:
            Performance grade (A-F)

        """
        if not scenario_win_rates:
            return "F"

        avg_win_rate = np.mean(list(scenario_win_rates.values()))

        if avg_win_rate >= 0.8:
            return "A"
        if avg_win_rate >= 0.6:
            return "B"
        if avg_win_rate >= 0.4:
            return "C"
        if avg_win_rate >= 0.25:
            return "D"
        return "F"

    def _print_evaluation_summary(self, summary: dict[str, Any]) -> None:
        """Print evaluation summary to console.

        Args:
            summary: Evaluation summary dictionary

        """
        print(f"\n📊 Evaluation Summary for {Path(summary['model_path']).name}")
        print(f"{'=' * 60}")

        overall = summary["overall_performance"]
        print(f"🏆 Overall Win Rate: {overall['overall_win_rate']:.1%}")
        print(f"🃏 Average Cards Left: {overall['overall_avg_cards_left']:.1f}")
        print(f"📈 Performance Grade: {overall['performance_grade']}")
        print(f"⏱️  Total Duration: {summary['evaluation_config']['total_duration_seconds']:.1f}s")
        print(f"🎮 Total Games: {overall['total_games']}")

        print("\n📋 Scenario Breakdown:")
        for scenario, win_rate in overall["scenario_win_rates"].items():
            scenario_display = scenario.replace("vs_", "").replace("_", " ").title()
            print(f"  {scenario_display}: {win_rate:.1%}")

        print(f"{'=' * 60}")

    def compare_models(
        self,
        model_paths: list[str],
        opponents: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compare multiple models against the same opponents.

        Args:
            model_paths: List of model paths to compare
            opponents: Opponent types for comparison
            **kwargs: Additional evaluation parameters

        Returns:
            Comparison results

        """
        if self.verbose:
            print(f"⚔️  Comparing {len(model_paths)} models")

        model_results = {}

        for model_path in model_paths:
            model_name = Path(model_path).stem
            if self.verbose:
                print(f"\n🔍 Evaluating {model_name}...")

            try:
                results = self.evaluate_model(model_path, opponents, **kwargs)
                model_results[model_name] = results
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to evaluate {model_name}: {e}")
                model_results[model_name] = {"error": str(e)}

        # Create comparison summary
        comparison = self._create_comparison_summary(model_results)

        if self.verbose:
            self._print_comparison_summary(comparison)

        return comparison

    def _create_comparison_summary(self, model_results: dict[str, Any]) -> dict[str, Any]:
        """Create comparison summary from model results.

        Args:
            model_results: Results for each model

        Returns:
            Comparison summary

        """
        comparison = {
            "models_compared": list(model_results.keys()),
            "comparison_metrics": {},
            "rankings": {},
            "model_results": model_results,
        }

        # Extract key metrics for comparison
        metrics = {}
        for model_name, results in model_results.items():
            if "error" in results:
                continue

            overall = results.get("overall_performance", {})
            metrics[model_name] = {
                "overall_win_rate": overall.get("overall_win_rate", 0),
                "overall_avg_cards_left": overall.get("overall_avg_cards_left", 13),
                "performance_grade": overall.get("performance_grade", "F"),
                "scenario_win_rates": overall.get("scenario_win_rates", {}),
            }

        comparison["comparison_metrics"] = metrics

        # Create rankings
        if metrics:
            # Rank by overall win rate
            sorted_by_winrate = sorted(
                metrics.items(),
                key=lambda x: x[1]["overall_win_rate"],
                reverse=True,
            )
            comparison["rankings"]["by_win_rate"] = [name for name, _ in sorted_by_winrate]

            # Rank by average cards left (lower is better)
            sorted_by_cards = sorted(
                metrics.items(),
                key=lambda x: x[1]["overall_avg_cards_left"],
            )
            comparison["rankings"]["by_efficiency"] = [name for name, _ in sorted_by_cards]

        return comparison

    def _print_comparison_summary(self, comparison: dict[str, Any]) -> None:
        """Print comparison summary to console.

        Args:
            comparison: Comparison summary

        """
        print("\n🏆 Model Comparison Results")
        print(f"{'=' * 60}")

        rankings = comparison.get("rankings", {})
        metrics = comparison.get("comparison_metrics", {})

        if "by_win_rate" in rankings:
            print("🥇 Win Rate Rankings:")
            for i, model_name in enumerate(rankings["by_win_rate"][:5], 1):
                win_rate = metrics[model_name]["overall_win_rate"]
                grade = metrics[model_name]["performance_grade"]
                print(f"  {i}. {model_name}: {win_rate:.1%} (Grade: {grade})")

        if "by_efficiency" in rankings:
            print("\n🎯 Efficiency Rankings (fewer cards left):")
            for i, model_name in enumerate(rankings["by_efficiency"][:5], 1):
                avg_cards = metrics[model_name]["overall_avg_cards_left"]
                print(f"  {i}. {model_name}: {avg_cards:.1f} cards avg")

        print(f"{'=' * 60}")


# Convenience functions
def evaluate_fixed_action_model(
    model_path: str,
    num_games: int = 100,
    opponents: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to evaluate a fixed action model.

    Args:
        model_path: Path to model file
        num_games: Number of games per evaluation
        opponents: Opponent types
        **kwargs: Additional arguments

    Returns:
        Evaluation results

    """
    evaluator = Evaluator(num_games=num_games, verbose=True)
    return evaluator.evaluate_model(model_path, opponents, **kwargs)


def compare_fixed_action_models(
    model_paths: list[str],
    num_games: int = 100,
    opponents: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to compare multiple fixed action models.

    Args:
        model_paths: List of model paths
        num_games: Number of games per evaluation
        opponents: Opponent types
        **kwargs: Additional arguments

    Returns:
        Comparison results

    """
    evaluator = Evaluator(num_games=num_games, verbose=True)
    return evaluator.compare_models(model_paths, opponents, **kwargs)
