"""Human agent that provides interactive gameplay through console interface."""

import sys
import numpy as np
from typing import Optional, Any
from .base_agent import BaseAgent
from ..core.card_utils import hand_array_to_strings, format_hand_array, hand_to_strings


class HumanAgent(BaseAgent):
    """Interactive human agent for playing Big Two via console interface."""

    def __init__(self, name: str = "Human"):
        super().__init__(name)
        self.env = None  # Will be set via set_env_reference()
        self.move_history = []  # Track recent moves for display

    def get_action(
        self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Get action from human player via console interface."""
        if self.env is None:
            raise RuntimeError(
                "HumanAgent requires env reference via set_env_reference()"
            )

        # Display current game state
        self._display_game_state()

        # Get legal moves from action mask
        if action_mask is None:
            raise RuntimeError(
                "HumanAgent requires action_mask to determine legal moves"
            )

        legal_action_indices = np.where(action_mask)[0]

        # Get the actual legal moves from the environment for the current player
        env_legal_moves = self.env.env.legal_moves(self.env.env.current_player)

        # Map action indices to actual moves - action_idx directly corresponds to env_legal_moves[action_idx]
        legal_moves = []
        for action_idx in legal_action_indices:
            if action_idx < len(env_legal_moves):
                move = env_legal_moves[action_idx]

                # Double-check that this move is actually legal
                if isinstance(move, np.ndarray):
                    if np.sum(move) == 0:  # Pass move
                        is_legal = (
                            self.env.env.last_play is not None
                        )  # Can only pass if there's a last play
                    else:
                        is_legal = self.env.env._beats(move)
                else:
                    # List format
                    if len(move) == 0:  # Pass move
                        is_legal = self.env.env.last_play is not None
                    else:
                        is_legal = self.env.env._beats(move)

                if is_legal:
                    legal_moves.append((action_idx, move))
                else:
                    print(
                        f"🚨 BUG DETECTED: Action {action_idx} appears in mask but is not legal!"
                    )
                    if isinstance(move, np.ndarray):
                        move_cards = hand_array_to_strings(move)
                        print(f"   Illegal move: {' '.join(move_cards)}")
                    else:
                        move_cards = hand_to_strings(move) if move else ["PASS"]
                        print(f"   Illegal move: {' '.join(move_cards)}")

                    if self.env.env.last_play:
                        last_cards = hand_array_to_strings(self.env.env.last_play[0])
                        print(f"   Must beat: {' '.join(last_cards)}")
                    else:
                        print(f"   Starting new trick")
            # Skip action indices that are out of bounds (shouldn't happen with proper action mask)

        # Display legal moves to user
        self._display_legal_moves(legal_moves)

        # Get user input
        return self._get_user_choice(legal_moves)

    def reset(self) -> None:
        """Reset agent state for new game."""
        pass  # Nothing to reset for human agent

    def set_env_reference(self, env: Any) -> None:
        """Store environment reference for accessing game state."""
        self.env = env

    def _display_game_state(self):
        """Display current game state to the human player."""
        print("\n" + "=" * 60)
        print("🃏 BIG TWO - HUMAN VS AI AGENTS")
        print("=" * 60)

        # Current player info
        current_player = self.env.env.current_player
        if current_player == 0:  # Assuming human is player 0
            print("🎯 YOUR TURN")
        else:
            print(f"🤖 Agent {current_player} is thinking...")

        # Last play information
        if self.env.env.last_play:
            last_play_array = self.env.env.last_play[0]
            last_cards = hand_array_to_strings(last_play_array)
            last_player = self.env.env.last_play[1]
            last_player_name = "YOU" if last_player == 0 else f"Agent {last_player}"

            # Identify hand type for display
            last_play_indices = np.where(last_play_array)[0]
            hand_type, _ = self.env.env._identify_hand_type(last_play_indices)
            if len(last_play_indices) == 1:
                type_display = "Single"
            elif len(last_play_indices) == 2:
                type_display = "Pair"
            elif len(last_play_indices) == 3:
                type_display = "Trips"
            elif len(last_play_indices) == 5:
                type_display = hand_type.replace("_", " ").title()
            else:
                type_display = f"{len(last_play_indices)}-card"

            print(
                f"Last play: {' '.join(last_cards)} ({type_display}) by {last_player_name}"
            )
        else:
            print("Last play: None (start new trick)")

        # Show pass information
        passes = self.env.env.passes_in_row
        if passes > 0:
            if passes == 1:
                print(f"⏭️  {passes} player has passed")
            else:
                print(f"⏭️  {passes} players have passed in a row")

        # Show if trick will reset
        if passes == self.env.env.num_players - 1:
            print("🔄 Next play will start a new trick!")

        # Show recent move history (last 3 moves)
        self._display_recent_moves()

        # Hand sizes for all players
        hand_sizes = [np.sum(hand) for hand in self.env.env.hands]
        print(
            f"Hand sizes: YOU={hand_sizes[0]}, Agent1={hand_sizes[1]}, Agent2={hand_sizes[2]}, Agent3={hand_sizes[3]}"
        )

        # Show human player's hand
        print(f"Your hand: {format_hand_array(self.env.env.hands[0])}")
        print("=" * 60)

    def _display_recent_moves(self):
        """Display the last 3 moves for context."""
        if not hasattr(self.env.env, "move_history") or not self.env.env.move_history:
            return

        print("\n📜 RECENT MOVES:")
        print("-" * 30)

        # Show last 3 moves (or all moves if fewer than 3)
        recent_moves = self.env.env.move_history[-3:]

        for i, (cards_array, player_idx, move_type) in enumerate(recent_moves):
            player_name = "YOU" if player_idx == 0 else f"Agent {player_idx}"

            if move_type == "Pass":
                print(f"  {player_name}: PASS")
            else:
                cards = hand_array_to_strings(cards_array)
                cards_str = " ".join(cards)
                print(f"  {player_name}: {cards_str} ({move_type})")

        print("-" * 30)

    def _display_legal_moves(self, legal_moves):
        """Display legal moves to the user."""
        print("\n📋 LEGAL MOVES:")
        print("-" * 40)

        for i, (action_idx, move) in enumerate(legal_moves):
            if move is None or (isinstance(move, np.ndarray) and np.sum(move) == 0):
                print(f"{i}: PASS")
            else:
                # Convert boolean array to card strings
                if isinstance(move, np.ndarray):
                    cards_str = " ".join(hand_array_to_strings(move))
                    move_indices = np.where(move)[0]
                    move_len = len(move_indices)
                else:
                    # Fallback for list format (shouldn't happen with current implementation)
                    cards_str = " ".join(hand_to_strings(move))
                    move_indices = move
                    move_len = len(move)

                # Identify hand type for display
                if move_len == 1:
                    type_display = "Single"
                elif move_len == 2:
                    type_display = "Pair"
                elif move_len == 3:
                    type_display = "Trips"
                elif move_len == 5:
                    hand_type, _ = self.env.env._identify_hand_type(move_indices)
                    type_display = hand_type.replace("_", " ").title()
                else:
                    type_display = f"{move_len}-card"

                print(f"{i}: {cards_str} ({type_display})")

        print("-" * 40)

    def _get_user_choice(self, legal_moves) -> int:
        """Get user's move choice with input validation."""
        while True:
            try:
                user_input = input(
                    f"\nEnter move number (0-{len(legal_moves) - 1}): "
                ).strip()

                # Handle special commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Thanks for playing!")
                    sys.exit(0)

                # Validate numeric input
                try:
                    choice = int(user_input)
                    if 0 <= choice < len(legal_moves):
                        action_idx, move = legal_moves[choice]

                        # Confirm the move
                        if move is None or (
                            isinstance(move, np.ndarray) and np.sum(move) == 0
                        ):
                            print("✅ You chose: PASS")
                        else:
                            if isinstance(move, np.ndarray):
                                cards_str = " ".join(hand_array_to_strings(move))
                            else:
                                cards_str = " ".join(hand_to_strings(move))
                            print(f"✅ You chose: {cards_str}")

                        return action_idx
                    else:
                        print(
                            f"❌ Invalid choice. Please enter a number between 0 and {len(legal_moves) - 1}"
                        )

                except ValueError:
                    print("❌ Please enter a valid number (or 'quit' to exit)")

            except KeyboardInterrupt:
                print("\n👋 Game interrupted. Thanks for playing!")
                sys.exit(0)
            except EOFError:
                print("\n👋 Input ended. Thanks for playing!")
                sys.exit(0)
