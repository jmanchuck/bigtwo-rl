"""Move quality-based reward system for strategic Big Two learning.

This module provides immediate feedback on move quality rather than sparse end-game rewards,
teaching agents strategic decision-making through dense reward signals.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from .base_reward import BaseReward


class MoveQualityReward(BaseReward):
    """
    Reward system that evaluates move quality immediately after each decision.
    
    Instead of sparse end-game rewards, this provides rich feedback on:
    - Card efficiency (playing low while keeping high)  
    - Hand type complexity (straights, flushes, etc.)
    - Timing appropriateness (when to play powerful combos)
    - Hand preservation (maintaining future options)
    
    This enables PPO to learn strategic reasoning through dense reward signals.
    """

    def __init__(
        self,
        card_efficiency_weight: float = 0.3,
        hand_type_weight: float = 0.2,
        timing_weight: float = 0.3,
        preservation_weight: float = 0.2,
        move_reward_scale: float = 0.2,
        game_reward_scale: float = 1.0,
    ):
        """
        Initialize move quality reward system.
        
        Args:
            card_efficiency_weight: Weight for card efficiency scoring
            hand_type_weight: Weight for hand type complexity scoring  
            timing_weight: Weight for timing appropriateness scoring
            preservation_weight: Weight for hand preservation scoring
            move_reward_scale: Scale factor for move quality rewards
            game_reward_scale: Scale factor for end-game rewards
        """
        self.card_efficiency_weight = card_efficiency_weight
        self.hand_type_weight = hand_type_weight
        self.timing_weight = timing_weight
        self.preservation_weight = preservation_weight
        self.move_reward_scale = move_reward_scale
        self.game_reward_scale = game_reward_scale
        
        # Validate weights sum to 1.0
        total_weight = (card_efficiency_weight + hand_type_weight + 
                       timing_weight + preservation_weight)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Card value mapping for Big Two (3=0, 4=1, ..., A=11, 2=12)
        self.card_values = np.array([
            val % 13 for val in range(52)  # Maps card index to Big Two value
        ])
        
        # Hand type scoring for different combinations
        self.hand_type_scores = {
            'single': 0.1,
            'pair': 0.3,
            'trip': 0.5, 
            'straight': 0.8,
            'flush': 0.9,
            'full_house': 0.95,
            'four_of_a_kind': 0.98,
            'straight_flush': 1.0,
            'pass': 0.0  # Will be evaluated separately
        }
        
        # Power cards (Aces and 2s) for efficiency calculations
        self.power_cards = set(range(44, 52))  # A♦,A♣,A♥,A♠,2♦,2♣,2♥,2♠

    def move_bonus(self, move_cards: List[int], game_context: Optional[Dict] = None) -> float:
        """
        Calculate immediate move quality reward.
        
        Args:
            move_cards: List of card indices played (empty for pass)
            game_context: Additional game state information
            
        Returns:
            float: Move quality reward (0.0 to move_reward_scale)
        """
        # Handle pass moves
        if not move_cards:
            return self._evaluate_pass_quality(game_context) * self.move_reward_scale
        
        # Extract game context with defaults
        if game_context is None:
            game_context = {}
            
        remaining_hand = game_context.get('remaining_hand', [])
        opponent_card_counts = game_context.get('opponent_card_counts', [10, 10, 10])
        game_phase = game_context.get('game_phase', 'MIDGAME')
        
        # Calculate quality factors
        efficiency_score = self._card_efficiency_score(move_cards, remaining_hand)
        hand_type_score = self._hand_type_score(move_cards)
        timing_score = self._timing_score(move_cards, opponent_card_counts, game_phase)
        preservation_score = self._hand_preservation_score(move_cards, remaining_hand)
        
        # Weighted combination
        total_quality = (
            efficiency_score * self.card_efficiency_weight +
            hand_type_score * self.hand_type_weight +
            timing_score * self.timing_weight +
            preservation_score * self.preservation_weight
        )
        
        return total_quality * self.move_reward_scale

    def _card_efficiency_score(self, played_cards: List[int], remaining_hand: List[int]) -> float:
        """
        Score based on card efficiency - playing low cards while keeping high ones.
        
        Returns value between 0.0 (inefficient) and 1.0 (highly efficient).
        """
        if not played_cards or not remaining_hand:
            return 0.5  # Neutral if insufficient information
            
        # Calculate average card values
        played_values = [self.card_values[card] for card in played_cards]
        remaining_values = [self.card_values[card] for card in remaining_hand]
        
        played_avg = np.mean(played_values)
        remaining_avg = np.mean(remaining_values)
        
        # Reward playing low while keeping high
        # Value difference ranges from -12 to +12, normalize to 0-1
        value_diff = remaining_avg - played_avg
        efficiency = max(0.0, min(1.0, (value_diff + 6) / 12.0))
        
        # Bonus for keeping power cards
        power_cards_kept = len([c for c in remaining_hand if c in self.power_cards])
        power_cards_played = len([c for c in played_cards if c in self.power_cards])
        
        if power_cards_kept > power_cards_played:
            efficiency = min(1.0, efficiency + 0.1)
            
        return efficiency

    def _hand_type_score(self, move_cards: List[int]) -> float:
        """
        Score based on hand type complexity.
        
        Returns value between 0.0 (simple) and 1.0 (most complex).
        """
        if not move_cards:
            return 0.0
            
        hand_type = self._identify_hand_type_simple(move_cards)
        return self.hand_type_scores.get(hand_type, 0.1)

    def _timing_score(self, move_cards: List[int], opponent_card_counts: List[int], 
                     game_phase: str) -> float:
        """
        Score based on timing appropriateness of the move.
        
        Returns value between 0.0 (poor timing) and 1.0 (excellent timing).
        """
        if not move_cards or not opponent_card_counts:
            return 0.5  # Neutral if insufficient information
            
        min_opponent_cards = min(opponent_card_counts) if opponent_card_counts else 10
        move_strength = len(move_cards)
        
        # Timing logic for powerful combinations (5+ cards)
        if move_strength >= 5:
            if min_opponent_cards >= 8:
                return 1.0  # Excellent - play big combos when opponents have many cards
            elif min_opponent_cards <= 3:
                return 0.2  # Poor - too late, opponents close to winning
            else:
                return 0.6  # Okay timing
        
        # Timing logic for medium combinations (2-4 cards)
        elif move_strength >= 2:
            if game_phase == 'ENDGAME' and min_opponent_cards <= 5:
                return 0.9  # Good - use pairs/trips to block in endgame
            else:
                return 0.7  # Generally okay
        
        # Single cards - generally neutral timing
        else:
            return 0.5

    def _hand_preservation_score(self, played_cards: List[int], remaining_hand: List[int]) -> float:
        """
        Score based on how well the move preserves future combination potential.
        
        Returns value between 0.0 (destroys options) and 1.0 (preserves options).
        """
        if not remaining_hand:
            return 1.0  # Perfect if hand is empty after move
            
        if not played_cards:
            return 1.0  # Pass preserves all options
            
        # Simple heuristic: count potential pairs, trips, and straights
        original_hand = played_cards + remaining_hand
        
        original_potential = self._count_combination_potential(original_hand)
        remaining_potential = self._count_combination_potential(remaining_hand)
        
        # Score based on how much potential was preserved
        if original_potential == 0:
            return 1.0  # No potential to lose
            
        preservation_ratio = remaining_potential / original_potential
        return min(1.0, preservation_ratio)

    def _count_combination_potential(self, hand: List[int]) -> int:
        """Count potential future combinations in a hand."""
        if not hand:
            return 0
            
        # Count by ranks for pairs/trips
        rank_counts = {}
        for card in hand:
            rank = card // 4  # Get rank (0-12)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        potential = 0
        
        # Count pair/trip potential
        for count in rank_counts.values():
            if count >= 2:
                potential += 1  # Can make pair
            if count >= 3:
                potential += 1  # Can make trip
            if count >= 4:
                potential += 2  # Four of a kind is very valuable
                
        # Count straight potential (simplified)
        unique_ranks = sorted(rank_counts.keys())
        consecutive_count = 1
        for i in range(1, len(unique_ranks)):
            if unique_ranks[i] == unique_ranks[i-1] + 1:
                consecutive_count += 1
                if consecutive_count >= 5:
                    potential += 2  # Potential straight
            else:
                consecutive_count = 1
                
        return potential

    def _evaluate_pass_quality(self, game_context: Optional[Dict]) -> float:
        """
        Evaluate the quality of passing instead of playing.
        
        Returns value between 0.0 (bad pass) and 1.0 (strategic pass).
        """
        if not game_context:
            return 0.3  # Neutral pass quality
            
        # Extract context
        remaining_hand = game_context.get('remaining_hand', [])
        opponent_card_counts = game_context.get('opponent_card_counts', [10, 10, 10])
        last_play_strength = game_context.get('last_play_strength', 1)
        
        # Strategic pass when:
        # 1. Have strong cards but opponents have many cards (save for later)
        # 2. Last play was strong and we want to preserve our hand
        # 3. In endgame with good position
        
        min_opponent_cards = min(opponent_card_counts) if opponent_card_counts else 10
        power_cards_count = len([c for c in remaining_hand if c in self.power_cards])
        
        pass_quality = 0.3  # Base pass quality
        
        # Bonus for strategic conservation
        if power_cards_count >= 2 and min_opponent_cards >= 6:
            pass_quality += 0.4  # Good to save power cards
            
        # Bonus for avoiding strong last plays
        if last_play_strength >= 5:
            pass_quality += 0.3  # Good to avoid challenging strong plays
            
        return min(1.0, pass_quality)

    def _identify_hand_type_simple(self, cards: List[int]) -> str:
        """
        Simple hand type identification for move quality scoring.
        
        This is a simplified version focused on rewarding combination complexity.
        """
        if not cards:
            return 'pass'
        elif len(cards) == 1:
            return 'single'
        elif len(cards) == 2:
            # Check if pair
            if cards[0] // 4 == cards[1] // 4:
                return 'pair'
            else:
                return 'single'  # Invalid 2-card play, treat as single
        elif len(cards) == 3:
            # Check if trip
            ranks = [card // 4 for card in cards]
            if len(set(ranks)) == 1:
                return 'trip'
            else:
                return 'single'  # Invalid, treat as single
        elif len(cards) == 5:
            # Simplified 5-card type detection
            ranks = sorted([card // 4 for card in cards])
            suits = [card % 4 for card in cards]
            
            # Check for straight
            is_straight = all(ranks[i] == ranks[i-1] + 1 for i in range(1, 5))
            
            # Check for flush  
            is_flush = len(set(suits)) == 1
            
            # Check for full house (3+2)
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            counts = sorted(rank_counts.values())
            is_full_house = counts == [2, 3]
            
            # Check for four of a kind
            is_four_of_kind = 4 in rank_counts.values()
            
            if is_straight and is_flush:
                return 'straight_flush'
            elif is_four_of_kind:
                return 'four_of_a_kind'
            elif is_full_house:
                return 'full_house'
            elif is_flush:
                return 'flush'
            elif is_straight:
                return 'straight'
            else:
                return 'single'  # Invalid 5-card hand
        else:
            return 'single'  # Unknown type

    def game_reward(
        self, 
        winner_player: int, 
        player_idx: int, 
        cards_left: int,
        all_cards_left: Optional[List[int]] = None
    ) -> float:
        """
        Minimal end-game reward since move quality provides most learning signal.
        
        Returns:
            float: Simple win/loss reward scaled by game_reward_scale
        """
        if winner_player == player_idx:
            return 1.0 * self.game_reward_scale
        else:
            # Small penalty based on cards left
            return -0.1 * cards_left * self.game_reward_scale

    def episode_bonus(
        self, 
        games_won: int, 
        total_games: int, 
        avg_cards_left: float
    ) -> float:
        """
        Minimal episode bonus since move quality provides dense learning.
        
        Returns:
            float: Small bonus for overall episode performance
        """
        win_rate = games_won / total_games if total_games > 0 else 0
        return (win_rate - 0.25) * 0.5  # Bonus for >25% win rate