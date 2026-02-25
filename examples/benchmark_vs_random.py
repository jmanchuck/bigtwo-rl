"""Benchmark a trained PPO model against 3 random or greedy agents.

Usage:
  ./.venv/bin/python examples/benchmark_vs_random.py --model models/foo/final_model.zip --opponent random --stochastic
"""

from __future__ import annotations

import argparse

import numpy as np

from bigtwo_rl.agents.ppo_agent import PPOAgent
from bigtwo_rl.agents.greedy_agent import GreedyAgent
from bigtwo_rl.agents.random_agent import RandomAgent
from bigtwo_rl.core.action import OFF_PASS, ActionMaskBuilder, BitsetFiveCardEngine, action_to_tuple
from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules
from bigtwo_rl.core.observation import EnhancedObservationBuilder


def play_one_game(
    ppo_agent: PPOAgent,
    opponents: list,
    seed: int,
) -> int:
    obs_dim = int(ppo_agent.model.observation_space.shape[0])
    if obs_dim != 412:
        raise ValueError(f"Unsupported observation size {obs_dim}; expected 412")

    game = ToyBigTwoFullRules(num_players=4, track_move_history=True)
    game.reset(seed=seed)

    obs_builder = EnhancedObservationBuilder()
    masker = ActionMaskBuilder(BitsetFiveCardEngine())

    for _ in range(500):
        if game.done:
            break

        player = game.current_player
        hand = game.get_player_hand(player)
        counts = game.get_player_card_counts()
        last_played = game.get_last_played_cards_encoded()
        is_first = (game.last_play is None) and all(c == 13 for c in counts)
        has_control = (game.last_play is None) and (not is_first)

        obs = obs_builder.build_observation(
            hand=hand,
            current_player=player,
            player_card_counts=counts,
            last_played_cards=last_played,
            passes=game.passes_in_row,
            is_first_play=is_first,
            move_history=getattr(game, "move_history", None),
            has_control=has_control,
            can_pass=(not is_first),
        )
        valid_ids = masker.full_mask_indices(
            hand,
            last_played,
            pass_allowed=not is_first,
            is_first_play=is_first,
            has_control=has_control,
        )
        action_mask = np.zeros(1365, dtype=bool)
        for idx in valid_ids:
            if 0 <= idx < 1365:
                action_mask[idx] = True

        if action_mask.any():
            if player == 0:
                action = ppo_agent.get_action(obs, action_mask)
            else:
                action = opponents[player - 1].get_action(obs, action_mask)
        else:
            action = OFF_PASS

        slots = [] if action == OFF_PASS else list(action_to_tuple(action))
        if not game.play_hand_move(player, slots):
            game.play_hand_move(player, [])

    cards_left = game.get_player_card_counts()
    winner = int(np.argmin(cards_left))
    if cards_left[winner] != 0:
        return -1
    return winner


def evaluate(model_path: str, games: int, strategy: str, seed: int, opponent: str, stochastic: bool) -> float:
    ppo = PPOAgent(model_path, deterministic=not stochastic, deterministic_strategy=strategy)
    if opponent == "greedy":
        opponents = [GreedyAgent("G1"), GreedyAgent("G2"), GreedyAgent("G3")]
    else:
        opponents = [RandomAgent("R1"), RandomAgent("R2"), RandomAgent("R3")]
    rng = np.random.default_rng(seed)

    wins = 0
    for _ in range(games):
        winner = play_one_game(ppo, opponents, int(rng.integers(1_000_000_000)))
        if winner == 0:
            wins += 1
    return wins / games


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model .zip")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling for PPO agent.")
    args = parser.parse_args()

    strategies = ("argmax", "nonpass_argmax")
    if args.stochastic:
        # Deterministic strategy does not apply in stochastic mode.
        strategies = ("argmax",)

    for strategy in strategies:
        wr = evaluate(
            args.model,
            games=args.games,
            strategy=strategy,
            seed=args.seed,
            opponent=args.opponent,
            stochastic=args.stochastic,
        )
        mode = "stochastic" if args.stochastic else f"deterministic[{strategy}]"
        print(f"mode={mode} opponent={args.opponent} win_rate={wr:.4f} games={args.games}")


if __name__ == "__main__":
    main()
