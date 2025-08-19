"""Different reward functions for Big Two training experiments."""

from abc import ABC, abstractmethod


class BaseReward(ABC):
    """Base class for custom reward functions with intermediate rewards."""

    @abstractmethod
    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """
        Calculate immediate reward for a player after each game completion.

        Args:
            winner_player: Index of winning player
            player_idx: Index of player to calculate reward for
            cards_left: Number of cards left for this player
            all_cards_left: List of cards left for all players (for ranking-based rewards)

        Returns:
            float: Immediate reward value
        """
        pass

    @abstractmethod
    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """
        Calculate bonus reward at episode end based on overall performance.

        Args:
            games_won: Number of games won in this episode
            total_games: Total games played in this episode
            avg_cards_left: Average cards remaining when losing games

        Returns:
            float: Episode bonus reward
        """
        pass


class DefaultReward(BaseReward):
    """Default reward structure with immediate game rewards."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """Immediate reward after each game."""
        if player_idx == winner_player:
            return 1.0  # Winner gets positive reward
        else:
            # Non-winners: penalty based on cards remaining
            if cards_left >= 10:
                return -1.0  # Large penalty for many cards
            elif cards_left >= 7:
                return cards_left * -0.1
            elif cards_left >= 3:
                return cards_left * -0.05
            elif cards_left >= 1:
                return cards_left * -0.02
            else:
                return 0

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """Small bonus for good episode performance."""
        win_rate = games_won / total_games if total_games > 0 else 0
        if win_rate > 0.6:
            return 0.5  # Bonus for >60% win rate
        return 0


class SparseReward(BaseReward):
    """Sparse reward - only win/loss, no card count penalty."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """Simple win/loss reward."""
        if player_idx == winner_player:
            return 1.0  # Win
        else:
            return -0.25  # Loss (reduced from -1 for better learning)

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """No episode bonus for sparse rewards."""
        return 0


class AggressivePenaltyReward(BaseReward):
    """Higher penalties for losing with many cards."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """Aggressive penalties for poor performance."""
        if player_idx == winner_player:
            return 2.0  # Higher win reward
        else:
            # Steeper penalties (scaled down for immediate rewards)
            if cards_left >= 8:
                return cards_left * -0.5
            elif cards_left >= 4:
                return cards_left * -0.3
            else:
                return cards_left * -0.1

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """Large bonus for avoiding penalties."""
        win_rate = games_won / total_games if total_games > 0 else 0
        if win_rate > 0.7:
            return 2.0  # Large bonus for high win rate
        elif avg_cards_left < 5.0:  # Good at minimizing cards when losing
            return 0.5
        return 0


class ProgressiveReward(BaseReward):
    """Reward for making progress (fewer cards = better reward)."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """Reward progress in reducing cards."""
        if player_idx == winner_player:
            return 1.5
        else:
            # Reward inversely proportional to cards left
            # 13 cards at start, so normalize
            progress_reward = (13 - cards_left) * 0.1
            return progress_reward - 0.5  # Base penalty + progress bonus

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """Bonus for consistent progress."""
        if avg_cards_left < 4.0:  # Very good at minimizing cards
            return 1.0
        elif avg_cards_left < 7.0:  # Decent progress
            return 0.3
        return 0


class RankingReward(BaseReward):
    """Reward based on final ranking among all players."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left):
        """Reward based on ranking."""
        if player_idx == winner_player:
            return 1.0

        # Rank players by cards left (fewer = better rank)
        sorted_players = sorted(enumerate(all_cards_left), key=lambda x: x[1])
        rank = next(i for i, (p, _) in enumerate(sorted_players) if p == player_idx)

        # Rank 0 = winner (handled above), 1 = 2nd place, etc.
        return 0.5 - rank * 0.2  # 2nd place gets 0.3, 3rd gets 0.1, last gets -0.1

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """Bonus for consistent ranking performance."""
        win_rate = games_won / total_games if total_games > 0 else 0
        # Bonus based on both wins and low card count when losing
        combined_score = win_rate + (1.0 - min(avg_cards_left / 10.0, 1.0))
        return max(0, combined_score - 1.0)


class FunctionReward(BaseReward):
    """Wrapper to use old function-based rewards with new API."""

    def __init__(self, reward_func):
        self.reward_func = reward_func

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """Convert function call to game reward."""
        if all_cards_left is not None and len(all_cards_left) > 0:
            return self.reward_func(winner_player, player_idx, cards_left, all_cards_left)
        else:
            return self.reward_func(winner_player, player_idx, cards_left)

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """No episode bonus for function-based rewards."""
        return 0


# Map of reward function names to reward classes
REWARD_FUNCTIONS = {
    "default": DefaultReward,
    "sparse": SparseReward,
    "aggressive_penalty": AggressivePenaltyReward,
    "progressive": ProgressiveReward,
    "ranking": RankingReward,
}


def get_reward_function(name="default"):
    """Get reward function by name, returns instantiated class."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function '{name}'. Available: {list(REWARD_FUNCTIONS.keys())}")
    return REWARD_FUNCTIONS[name]()


def list_reward_functions():
    """List all available reward function names."""
    return list(REWARD_FUNCTIONS.keys())


# Legacy function-based rewards (for backward compatibility)
def default_reward(winner_player, player_idx, cards_left):
    """Legacy default reward function."""
    return DefaultReward().game_reward(winner_player, player_idx, cards_left)

def sparse_reward(winner_player, player_idx, cards_left):
    """Legacy sparse reward function."""
    return SparseReward().game_reward(winner_player, player_idx, cards_left)

def aggressive_penalty_reward(winner_player, player_idx, cards_left):
    """Legacy aggressive penalty reward function."""
    return AggressivePenaltyReward().game_reward(winner_player, player_idx, cards_left)

def progressive_reward(winner_player, player_idx, cards_left):
    """Legacy progressive reward function."""
    return ProgressiveReward().game_reward(winner_player, player_idx, cards_left)

def ranking_reward(winner_player, player_idx, cards_left, all_cards_left):
    """Legacy ranking reward function."""
    return RankingReward().game_reward(winner_player, player_idx, cards_left, all_cards_left)
