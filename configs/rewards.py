"""Different reward functions for Big Two training experiments."""

def default_reward(winner_player, player_idx, cards_left):
    """Current reward structure from bigtwo.py."""
    if player_idx == winner_player:
        return 5  # Winner gets massive reward
    else:
        # Non-winners: reward based on cards remaining (nonlinear)
        if cards_left >= 10:
            return cards_left * -3  # Hugely negative for 10+ cards
        elif cards_left >= 5:
            return cards_left * -1.5  # Negative for 5-9 cards
        elif cards_left >= 2:
            return cards_left * -1  # Small penalty for 1-4 cards
        else:
            return 0

def sparse_reward(winner_player, player_idx, cards_left):
    """Sparse reward - only win/loss, no card count penalty."""
    if player_idx == winner_player:
        return 1  # Win
    else:
        return -1  # Loss

def aggressive_penalty_reward(winner_player, player_idx, cards_left):
    """Higher penalties for losing with many cards."""
    if player_idx == winner_player:
        return 10  # Higher win reward
    else:
        # Much steeper penalties
        if cards_left >= 8:
            return cards_left * -5
        elif cards_left >= 4:
            return cards_left * -3
        else:
            return cards_left * -1

def progressive_reward(winner_player, player_idx, cards_left):
    """Reward for making progress (fewer cards = better reward)."""
    if player_idx == winner_player:
        return 5
    else:
        # Reward inversely proportional to cards left
        # 13 cards at start, so normalize
        progress_reward = (13 - cards_left) * 0.2
        return progress_reward - 2  # Base penalty + progress bonus

def ranking_reward(winner_player, player_idx, cards_left, all_cards_left):
    """Reward based on final ranking among all players."""
    if player_idx == winner_player:
        return 3
    
    # Rank players by cards left (fewer = better rank)
    sorted_players = sorted(enumerate(all_cards_left), key=lambda x: x[1])
    rank = next(i for i, (p, _) in enumerate(sorted_players) if p == player_idx)
    
    # Rank 0 = winner (handled above), 1 = 2nd place, etc.
    return 2 - rank  # 2nd place gets 1, 3rd gets 0, last gets negative

# Map of reward function names to functions
REWARD_FUNCTIONS = {
    "default": default_reward,
    "sparse": sparse_reward,
    "aggressive_penalty": aggressive_penalty_reward,
    "progressive": progressive_reward,
    "ranking": ranking_reward,
}

def get_reward_function(name="default"):
    """Get reward function by name."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function '{name}'. Available: {list(REWARD_FUNCTIONS.keys())}")
    return REWARD_FUNCTIONS[name]

def list_reward_functions():
    """List all available reward function names."""
    return list(REWARD_FUNCTIONS.keys())