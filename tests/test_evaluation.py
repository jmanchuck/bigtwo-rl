"""Test evaluation system with numpy arrays."""

from bigtwo_rl.agents import GreedyAgent, RandomAgent
from bigtwo_rl.evaluation import Evaluator, Tournament


def test_evaluator_with_numpy():
    """Test that evaluator works with numpy array representation."""
    print("Testing evaluator with numpy arrays...")

    evaluator = Evaluator(num_games=3)
    agent = RandomAgent("TestAgent")

    results = evaluator.evaluate_agent(agent)

    # Verify structure
    assert "players" in results
    assert "wins" in results
    assert "win_rates" in results
    assert "games_played" in results
    assert results["games_played"] == 3
    assert len(results["players"]) == 4  # Agent + 3 opponents

    print("✓ Evaluator test passed!")


def test_tournament_with_numpy():
    """Test that tournament works with numpy array representation."""
    print("Testing tournament with numpy arrays...")

    agents = [
        RandomAgent("Random-1"),
        RandomAgent("Random-2"),
        GreedyAgent("Greedy-1"),
        GreedyAgent("Greedy-2"),
    ]

    tournament = Tournament(agents)
    results = tournament.run(num_games=2)

    # Verify structure
    assert "agent_stats" in results
    assert "matchup_results" in results
    assert "tournament_summary" in results

    # Check that all agents have stats
    for agent in agents:
        assert agent.name in results["agent_stats"]
        assert "total_wins" in results["agent_stats"][agent.name]
        assert "win_rate" in results["agent_stats"][agent.name]

    print("✓ Tournament test passed!")


def test_card_counting_with_numpy():
    """Test that card counting works correctly with numpy arrays."""
    print("Testing card counting consistency...")

    import numpy as np

    from bigtwo_rl.core.observation_builder import strategic_observation
    from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper

    env = BigTwoRLWrapper(
        observation_config=strategic_observation(),
        num_players=4,
        games_per_episode=1,
    )
    env.reset()

    # Verify total cards is correct
    total_cards = sum(np.sum(env.env.hands[p]) for p in range(4))
    assert total_cards == 52, f"Expected 52 cards, got {total_cards}"

    # Verify each player has 13 cards initially
    for p in range(4):
        cards = np.sum(env.env.hands[p])
        assert cards == 13, f"Player {p} has {cards} cards, expected 13"

    print("✓ Card counting test passed!")


if __name__ == "__main__":
    test_evaluator_with_numpy()
    test_tournament_with_numpy()
    test_card_counting_with_numpy()
    print("\n✓ All evaluation tests passed!")
