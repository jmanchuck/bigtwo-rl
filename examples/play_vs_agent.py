#!/usr/bin/env python3
"""CLI interface to play Big Two against trained agent."""

import sys
import os
from stable_baselines3 import PPO
from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules
from bigtwo_rl.core.card_utils import hand_to_strings, format_hand, hand_array_to_strings, format_hand_array
import numpy as np


class HumanVsAgentGame:
    def __init__(self, model_path):
        self.env = ToyBigTwoFullRules(num_players=4)  # Use 4 players like training
        self.model = PPO.load(model_path)
        self.human_player = 0  # Human is player 0

    def display_game_state(self):
        """Show current game state."""
        print("\n" + "=" * 50)
        current_player_name = (
            "YOU" if self.env.current_player == self.human_player else f"AGENT {self.env.current_player}"
        )
        print(f"Current player: {current_player_name}")

        if self.env.last_play:
            last_play_cards = np.where(self.env.last_play[0])[0]
            last_cards = hand_to_strings(last_play_cards)
            last_player = "YOU" if self.env.last_play[1] == self.human_player else f"AGENT {self.env.last_play[1]}"
            print(f"Last play: {' '.join(last_cards)} (by {last_player})")
        else:
            print("Last play: None (start new trick)")

        # Show all hand sizes
        hand_sizes = [np.sum(hand) for hand in self.env.hands]
        print(
            f"Hand sizes: YOU={hand_sizes[0]}, AGENT1={hand_sizes[1]}, AGENT2={hand_sizes[2]}, AGENT3={hand_sizes[3]}"
        )
        print(f"Your hand: {format_hand_array(self.env.hands[self.human_player])}")
        print("=" * 50)

    def get_human_move(self):
        """Get move from human player."""
        legal_moves = self.env.legal_moves(self.human_player)

        print("\nLegal moves:")
        for i, move in enumerate(legal_moves):
            if not move:
                print(f"{i}: PASS")
            else:
                cards_str = " ".join(hand_to_strings(move))
                # Identify hand type for display
                hand_type, _ = self.env._identify_hand_type(move)
                if len(move) == 1:
                    print(f"{i}: {cards_str} (Single)")
                elif len(move) == 2:
                    print(f"{i}: {cards_str} (Pair)")
                elif len(move) == 3:
                    print(f"{i}: {cards_str} (Trips)")
                elif len(move) == 5:
                    hand_type_display = hand_type.replace("_", " ").title()
                    print(f"{i}: {cards_str} ({hand_type_display})")
                else:
                    print(f"{i}: {cards_str}")

        while True:
            try:
                user_input = input(f"\nEnter move number (0-{len(legal_moves)-1}): ").strip()

                try:
                    move_idx = int(user_input)
                    if 0 <= move_idx < len(legal_moves):
                        return move_idx
                    else:
                        print(f"Invalid move number. Choose 0-{len(legal_moves)-1}")
                except ValueError:
                    print("Please enter a number.")

            except KeyboardInterrupt:
                print("\nGame interrupted.")
                sys.exit(0)

    def get_agent_move(self, player_id):
        """Get move from trained agent for given player."""
        # Get raw observation from current game state
        raw_obs = self.env._get_obs()

        # Convert to 109-dim vector like our RL wrapper
        import numpy as np

        hand_binary = raw_obs["hand"].astype(np.float32)
        last_play_binary = raw_obs["last_play"].astype(np.float32)
        hand_sizes = np.array([np.sum(h) for h in self.env.hands], dtype=np.float32)
        last_play_exists = np.array([raw_obs["last_play_exists"]], dtype=np.float32)
        obs = np.concatenate([hand_binary, last_play_binary, hand_sizes, last_play_exists])

        action, _ = self.model.predict(obs, deterministic=True)

        # Convert action to move index
        legal_moves = self.env.legal_moves(player_id)
        if action >= len(legal_moves):
            return 0  # Fallback to first legal move
        return int(action)

    def play_game(self):
        """Play a full game."""
        print("🃏 Big Two: Human vs 3 Agents (FULL RULES)")
        print("Card format: AS=Ace of Spades, KH=King of Hearts, TD=Ten of Diamonds")
        print("Ranks: 3 < 4 < 5 < 6 < 7 < 8 < 9 < T < J < Q < K < A < 2")
        print("Suits: D < C < H < S")
        print(
            "Play types: Singles, Pairs, Trips, 5-card hands (Straight/Flush/Full House/Four of a Kind/Straight Flush)"
        )
        print("🎮 Select moves by entering the number (e.g., '0' for first option)")

        obs = self.env.reset(seed=None)

        while not self.env.done:
            self.display_game_state()

            current_player = self.env.current_player

            if current_player == self.human_player:  # Human turn
                move_idx = self.get_human_move()
                move = self.env.legal_moves(current_player)[move_idx]
                if move:
                    cards_str = " ".join(hand_to_strings(move))
                    hand_type, _ = self.env._identify_hand_type(move)
                    if len(move) == 5:
                        hand_type_display = hand_type.replace("_", " ").title()
                        print(f"\nYou played: {cards_str} ({hand_type_display})")
                    else:
                        print(f"\nYou played: {cards_str}")
                else:
                    print("\nYou passed.")
            else:  # Agent turn
                print(f"\nAgent {current_player} is thinking...")
                move_idx = self.get_agent_move(current_player)
                move = self.env.legal_moves(current_player)[move_idx]
                if move:
                    cards_str = " ".join(hand_to_strings(move))
                    hand_type, _ = self.env._identify_hand_type(move)
                    if len(move) == 5:
                        hand_type_display = hand_type.replace("_", " ").title()
                        print(f"Agent {current_player} played: {cards_str} ({hand_type_display})")
                    else:
                        print(f"Agent {current_player} played: {cards_str}")
                else:
                    print(f"Agent {current_player} passed.")

            obs, rewards, done, info = self.env.step(move_idx)

        # Game over
        print("\n" + "=" * 50)
        print("🎉 GAME OVER!")

        # Find who won (who has 0 cards)
        winner = None
        for i, hand in enumerate(self.env.hands):
            if np.sum(hand) == 0:
                winner = i
                break

        if winner == self.human_player:
            print("🏆 YOU WON!")
        elif winner is not None:
            print(f"🤖 AGENT {winner} WON!")
        else:
            print("Game ended unexpectedly.")

        hand_sizes = [np.sum(hand) for hand in self.env.hands]
        print(
            f"Final hand sizes - YOU: {hand_sizes[0]}, AGENT1: {hand_sizes[1]}, AGENT2: {hand_sizes[2]}, AGENT3: {hand_sizes[3]}"
        )
        print("=" * 50)


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/play_vs_agent.py <MODEL_DIR or MODEL_FILE>")
        print("  - If a directory is provided, the script will look for 'best_model.zip' then 'final_model.zip'.")
        sys.exit(1)

    user_arg = sys.argv[1]

    # Resolve model file from directory or direct file path
    if os.path.isdir(user_arg):
        candidate_paths = [
            os.path.join(user_arg, "best_model.zip"),
            os.path.join(user_arg, "final_model.zip"),
        ]
        model_path = None
        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                model_path = candidate
                break
        if model_path is None:
            # Fallback: any .zip in directory
            zips = [f for f in os.listdir(user_arg) if f.endswith(".zip")]
            if zips:
                model_path = os.path.join(user_arg, sorted(zips)[0])
        if model_path is None:
            print(f"No model file found in directory: {user_arg}")
            print("Expected 'best_model.zip' or 'final_model.zip'.")
            sys.exit(1)
    else:
        # User provided a specific file path
        model_path = user_arg
        if not os.path.isfile(model_path):
            print(f"Model file not found: {model_path}")
            sys.exit(1)

    try:
        game = HumanVsAgentGame(model_path)
        game.play_game()
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Run training first, then pass the model directory (e.g., ./models/<run_name>)")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
