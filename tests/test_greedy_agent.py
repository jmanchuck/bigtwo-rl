import numpy as np

from bigtwo_rl.agents.greedy_agent import GreedyAgent


def test_greedy_prefers_lowest_non_pass_legal_action() -> None:
    agent = GreedyAgent()
    mask = np.zeros(1365, dtype=bool)
    mask[0] = True
    mask[12] = True
    mask[25] = True
    assert agent.get_action(np.zeros(168, dtype=np.float32), mask) == 12


def test_greedy_falls_back_to_pass_when_only_pass_is_legal() -> None:
    agent = GreedyAgent()
    mask = np.zeros(1365, dtype=bool)
    mask[0] = True
    assert agent.get_action(np.zeros(168, dtype=np.float32), mask) == 0
