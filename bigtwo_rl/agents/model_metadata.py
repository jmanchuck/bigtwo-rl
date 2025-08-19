"""Model metadata system for observation configuration compatibility."""

import json
import os
from typing import Dict, Any, Optional
from ..core.observation_builder import ObservationConfig


class ModelMetadata:
    """Handles saving and loading model metadata for observation compatibility."""

    @staticmethod
    def save_metadata(
        model_dir: str,
        observation_config: ObservationConfig,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Save model metadata including observation configuration."""
        metadata = {
            "observation_config": {
                "include_hand": observation_config.include_hand,
                "include_last_play": observation_config.include_last_play,
                "include_hand_sizes": observation_config.include_hand_sizes,
                "include_played_cards": observation_config.include_played_cards,
                "include_remaining_deck": observation_config.include_remaining_deck,
                "include_cards_by_player": observation_config.include_cards_by_player,
                "include_last_play_exists": observation_config.include_last_play_exists,
                "include_game_phase": observation_config.include_game_phase,
                "include_turn_position": observation_config.include_turn_position,
                "include_trick_history": observation_config.include_trick_history,
                "include_pass_history": observation_config.include_pass_history,
                "include_play_patterns": observation_config.include_play_patterns,
                "include_power_cards_remaining": observation_config.include_power_cards_remaining,
                "include_hand_type_capabilities": observation_config.include_hand_type_capabilities,
                "_total_size": observation_config._total_size,
            }
        }

        if additional_info:
            metadata.update(additional_info)

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_metadata(model_dir: str) -> Dict[str, Any]:
        """Load model metadata."""
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def load_observation_config(model_dir: str) -> Optional[ObservationConfig]:
        """Load observation configuration from model metadata."""
        metadata = ModelMetadata.load_metadata(model_dir)
        obs_config_data = metadata.get("observation_config")

        if obs_config_data:
            # Reconstruct ObservationConfig from saved data
            config = ObservationConfig(
                include_hand=obs_config_data.get("include_hand", True),
                include_last_play=obs_config_data.get("include_last_play", True),
                include_hand_sizes=obs_config_data.get("include_hand_sizes", True),
                include_played_cards=obs_config_data.get("include_played_cards", False),
                include_remaining_deck=obs_config_data.get(
                    "include_remaining_deck", False
                ),
                include_cards_by_player=obs_config_data.get(
                    "include_cards_by_player", False
                ),
                include_last_play_exists=obs_config_data.get(
                    "include_last_play_exists", True
                ),
                include_game_phase=obs_config_data.get("include_game_phase", False),
                include_turn_position=obs_config_data.get(
                    "include_turn_position", False
                ),
                include_trick_history=obs_config_data.get(
                    "include_trick_history", False
                ),
                include_pass_history=obs_config_data.get("include_pass_history", False),
                include_play_patterns=obs_config_data.get(
                    "include_play_patterns", False
                ),
                include_power_cards_remaining=obs_config_data.get(
                    "include_power_cards_remaining", False
                ),
                include_hand_type_capabilities=obs_config_data.get(
                    "include_hand_type_capabilities", False
                ),
            )
            return config

        return None
