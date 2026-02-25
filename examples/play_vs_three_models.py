"""Play Big Two as a human against three trained PPO agents.

Usage:
  MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python examples/play_vs_three_models.py \
    --model-1 models/greedy_robust_long/checkpoint_1950000.zip \
    --model-2 models/greedy_robust_long/checkpoint_1050000.zip \
    --model-3 models/greedy_robust_long/checkpoint_2000000.zip
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from bigtwo_rl.agents.ppo_agent import PPOAgent
from bigtwo_rl.core.action import OFF_PASS, ActionMaskBuilder, BitsetFiveCardEngine, action_to_tuple
from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules
from bigtwo_rl.core.cards import card_to_string
from bigtwo_rl.core.observation import EnhancedObservationBuilder


def _resolve_model_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        candidates = [
            os.path.join(path_or_dir, "best_by_primary.zip"),
            os.path.join(path_or_dir, "best_by_greedy.zip"),
            os.path.join(path_or_dir, "best_by_random.zip"),
            os.path.join(path_or_dir, "best_model.zip"),
            os.path.join(path_or_dir, "final_model.zip"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        zip_files = sorted(f for f in os.listdir(path_or_dir) if f.endswith(".zip"))
        if zip_files:
            return os.path.join(path_or_dir, zip_files[0])
        raise FileNotFoundError(f"No .zip model found in directory: {path_or_dir}")
    if os.path.isfile(path_or_dir):
        return path_or_dir
    raise FileNotFoundError(f"Model file not found: {path_or_dir}")


def _describe_action(action_id: int, hand) -> str:
    if action_id == OFF_PASS:
        return "PASS"
    slots = list(action_to_tuple(action_id))
    cards: list[str] = []
    for slot in slots:
        if 0 <= slot < len(hand.card) and not hand.played[slot]:
            cards.append(card_to_string(hand.card[slot]))
    if not cards:
        return "INVALID"
    return " ".join(cards)


def _print_state(game: ToyBigTwoFullRules, hand) -> None:
    counts = game.get_player_card_counts()
    current_cards = [
        card_to_string(hand.card[i]) for i in range(len(hand.card)) if not hand.played[i]
    ]
    print("\n" + "=" * 70)
    print("Your turn (Player 0)")
    print(f"Card counts: You={counts[0]} | AI-1={counts[1]} | AI-2={counts[2]} | AI-3={counts[3]}")
    if game.last_play is None:
        print("Last play: None (you can lead any legal move)")
    else:
        last_mask, last_player = game.last_play
        last_cards = [card_to_string((c // 4 << 2) | (c % 4)) for c in np.where(last_mask)[0]]
        print(f"Last play by Player {last_player}: {' '.join(last_cards)}")
    print(f"Passes in row: {game.passes_in_row}")
    print(f"Your hand ({len(current_cards)}): {' '.join(current_cards)}")


def _choose_human_action(legal_action_ids: np.ndarray, hand, game: ToyBigTwoFullRules) -> int:
    _print_state(game, hand)
    print("\nLegal moves:")
    for i, action_id in enumerate(legal_action_ids):
        print(f"  {i:3d}: {_describe_action(int(action_id), hand)}")

    while True:
        raw = input("\nChoose move index (or 'q' to quit): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            idx = int(raw)
            if 0 <= idx < len(legal_action_ids):
                return int(legal_action_ids[idx])
            print(f"Enter an integer between 0 and {len(legal_action_ids) - 1}.")
        except ValueError:
            print("Enter a valid integer.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-1", required=True, help="Model file or directory for AI-1")
    parser.add_argument("--model-2", required=True, help="Model file or directory for AI-2")
    parser.add_argument("--model-3", required=True, help="Model file or directory for AI-3")
    parser.add_argument("--seed", type=int, default=None, help="Optional game seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PPO actions for AI opponents (default: stochastic)",
    )
    args = parser.parse_args()

    try:
        model_paths = [
            _resolve_model_path(args.model_1),
            _resolve_model_path(args.model_2),
            _resolve_model_path(args.model_3),
        ]
    except FileNotFoundError as exc:
        print(f"Failed to locate model: {exc}")
        sys.exit(1)

    print("Big Two: Human vs 3 trained agents")
    print(f"AI-1: {model_paths[0]}")
    print(f"AI-2: {model_paths[1]}")
    print(f"AI-3: {model_paths[2]}")
    print(f"AI policy mode: {'deterministic' if args.deterministic else 'stochastic'}")

    opponents = [
        PPOAgent(model_paths[0], "AI-1", deterministic=args.deterministic),
        PPOAgent(model_paths[1], "AI-2", deterministic=args.deterministic),
        PPOAgent(model_paths[2], "AI-3", deterministic=args.deterministic),
    ]

    game = ToyBigTwoFullRules(num_players=4, track_move_history=True)
    game.reset(seed=args.seed)
    obs_builder = EnhancedObservationBuilder()
    masker = ActionMaskBuilder(BitsetFiveCardEngine())

    try:
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

            legal_action_ids = np.where(action_mask)[0]
            if len(legal_action_ids) == 0:
                action = OFF_PASS
            elif player == 0:
                action = _choose_human_action(legal_action_ids, hand, game)
            else:
                action = opponents[player - 1].get_action(obs, action_mask)
                print(f"Player {player} plays: {_describe_action(action, hand)}")

            slots = [] if action == OFF_PASS else list(action_to_tuple(action))
            if not game.play_hand_move(player, slots):
                # Defensive fallback to pass if a move fails due to stale state.
                game.play_hand_move(player, [])

        cards_left = game.get_player_card_counts()
        winner = int(np.argmin(cards_left))
        print("\n" + "=" * 70)
        print("Game over")
        print(f"Winner: Player {winner} ({'You' if winner == 0 else f'AI-{winner}'})")
        print(f"Cards left: You={cards_left[0]}, AI-1={cards_left[1]}, AI-2={cards_left[2]}, AI-3={cards_left[3]}")
    except KeyboardInterrupt:
        print("\nExited game.")


if __name__ == "__main__":
    main()
