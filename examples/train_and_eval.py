import os
from pathlib import Path

import numpy as np

from bigtwo_rl.agents.greedy_agent import GreedyAgent
from bigtwo_rl.agents.ppo_agent import PPOAgent
from bigtwo_rl.training.trainer import Trainer


def train_model(total_timesteps: int = 10000) -> Path:
    trainer = Trainer()
    model, model_dir = trainer.train(
        total_timesteps=total_timesteps,
        model_name="quick_train_eval",
        eval_freq=max(total_timesteps // 5, 1000),
        verbose=1,
    )
    # Return path to the final model zip file that PPOAgent expects
    model_zip = Path(model_dir) / "final_model.zip"
    if not model_zip.exists():
        # stable-baselines3 .save(".../final_model") writes without .zip in our trainer
        # But PPO.load expects a .zip; SB3 automatically adds .zip. Ensure we have it.
        alt = Path(model_dir) / "final_model"
        if alt.exists():
            model_zip = alt
    return model_zip


def play_one_game(agents: list, seed: int | None = None) -> int:
    from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules

    game = ToyBigTwoFullRules(num_players=4, track_move_history=True)
    if seed is not None:
        game.reset(seed=seed)
    else:
        game.reset()

    # Play until done
    while not game.done:
        player = game.current_player
        # Build observation similar to BigTwoWrapper
        from bigtwo_rl.core.observation import EnhancedObservationBuilder

        if not hasattr(game, "_obs_builder"):
            game._obs_builder = EnhancedObservationBuilder()

        hand = game.get_player_hand(player)
        counts = game.get_player_card_counts()
        last_played = game.get_last_played_cards_encoded()
        passes = game.passes_in_row
        is_first = (game.last_play is None) and all(c == 13 for c in counts)

        obs = game._obs_builder.build_observation(
            hand=hand,
            current_player=player,
            player_card_counts=counts,
            last_played_cards=last_played,
            passes=passes,
            is_first_play=is_first,
            move_history=getattr(game, "move_history", None),
            has_control=(game.last_play is None) and (not is_first),
            can_pass=(not is_first),
        )

        # Build action mask using the same action masker as wrapper
        from bigtwo_rl.core.action import ActionMaskBuilder, BitsetFiveCardEngine

        if not hasattr(game, "_action_masker"):
            game._action_masker = ActionMaskBuilder(BitsetFiveCardEngine())

        pass_allowed = not is_first
        mask_indices = game._action_masker.full_mask_indices(
            hand,
            last_played,
            pass_allowed=pass_allowed,
            is_first_play=is_first,
            has_control=(game.last_play is None) and (not is_first),
        )
        action_mask = np.zeros(1365, dtype=bool)
        for idx in mask_indices:
            if 0 <= idx < 1365:
                action_mask[idx] = True

        action = agents[player].get_action(obs, action_mask)

        # Translate action id to slot indices and play
        from bigtwo_rl.core.action import OFF_PASS, action_to_tuple

        if action == OFF_PASS:
            slot_indices = []
        else:
            try:
                slot_indices = list(action_to_tuple(action))
            except ValueError:
                slot_indices = []

        ok = game.play_hand_move(player, slot_indices)
        if not ok:
            # Fallback to pass
            game.play_hand_move(player, [])

    # Determine winner (player with 0 cards)
    cards_left = game.get_player_card_counts()
    for i, c in enumerate(cards_left):
        if c == 0:
            return i
    return -1


def evaluate(model_path: Path, games: int = 100) -> float:
    # Our trained PPO agent will be Player 0, others are greedy
    ppo = PPOAgent(str(model_path), name="PPO-Agent", deterministic=True)
    agents = [ppo, GreedyAgent("Greedy-1"), GreedyAgent("Greedy-2"), GreedyAgent("Greedy-3")]

    wins = 0
    for g in range(games):
        winner = play_one_game(agents)
        if winner == 0:
            wins += 1
        # Reset agents if needed
        for a in agents:
            a.reset()

        if (g + 1) % 10 == 0:
            print(f"Played {g + 1}/{games} games; current win rate: {wins / (g + 1):.3f}")

    return wins / games


def main():
    # Train briefly, then evaluate
    total_timesteps = int(os.environ.get("BT_TRAIN_STEPS", "10000"))
    model_path = train_model(total_timesteps)
    print(f"Model saved at: {model_path}")

    win_rate = evaluate(model_path, games=100)
    print(f"PPO agent win rate vs 3 Greedy agents over 100 games: {win_rate:.3f}")


if __name__ == "__main__":
    main()
