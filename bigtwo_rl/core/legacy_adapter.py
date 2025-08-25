"""Legacy adapter for transitioning between old and new action systems.

This module provides compatibility layers to help transition from the dynamic
action space system to the fixed action space system.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from .action_system import BigTwoActionSystem
from .action_space import HandType
from .rl_wrapper import BigTwoRLWrapper
from .fixed_action_wrapper import FixedActionBigTwoWrapper
from .card_mapping import ActionTranslator


class LegacyActionAdapter:
    """Adapter to help transition from old to new action system."""
    
    def __init__(self):
        self.action_system = BigTwoActionSystem()
        self.translator = ActionTranslator()
        
    def convert_old_action_to_new(self, old_action_index: int, legal_moves: List[np.ndarray], player_hand: np.ndarray) -> int:
        """Convert old dynamic action index to new fixed action ID.
        
        Args:
            old_action_index: Index into legal_moves list
            legal_moves: List of legal moves from game engine
            player_hand: Player's 52-card hand array
            
        Returns:
            Fixed action ID from 0-1364
        """
        return self.translator.legacy_action_to_fixed_action(old_action_index, legal_moves, player_hand)
        
    def convert_new_action_to_old(self, action_id: int, legal_moves: List[np.ndarray], player_hand: np.ndarray) -> int:
        """Convert new fixed action ID to old dynamic index.
        
        Args:
            action_id: Fixed action ID from 0-1364
            legal_moves: List of legal moves from game engine  
            player_hand: Player's 52-card hand array
            
        Returns:
            Legacy action index
        """
        return self.translator.fixed_action_to_legacy_action(action_id, legal_moves, player_hand)
    
    def validate_action_compatibility(self, old_action: int, new_action: int, legal_moves: List[np.ndarray], player_hand: np.ndarray) -> bool:
        """Validate that old and new actions produce the same game move.
        
        Args:
            old_action: Legacy action index
            new_action: Fixed action ID
            legal_moves: List of legal moves
            player_hand: Player's hand
            
        Returns:
            True if actions are equivalent
        """
        try:
            # Convert old action to game move
            if old_action >= len(legal_moves):
                old_move = np.zeros(52, dtype=bool)  # Pass
            else:
                old_move = legal_moves[old_action]
                
            # Convert new action to game move
            new_move = self.action_system.translate_action_to_game_move(new_action, player_hand)
            
            # Compare moves
            return np.array_equal(old_move, new_move)
            
        except Exception:
            return False


class CompatibilityWrapper:
    """Wrapper that can work with both old and new action formats.
    
    This is a factory class that creates the appropriate wrapper based on
    the use_fixed_actions flag.
    """
    
    @staticmethod
    def create_wrapper(use_fixed_actions: bool = True, **kwargs) -> Union[BigTwoRLWrapper, FixedActionBigTwoWrapper]:
        """Create wrapper based on action space preference.
        
        Args:
            use_fixed_actions: If True, use fixed action space (1365 actions)
                              If False, use legacy dynamic action space
            **kwargs: Arguments passed to wrapper constructor
            
        Returns:
            Appropriate wrapper instance
        """
        if use_fixed_actions:
            return FixedActionBigTwoWrapper(**kwargs)
        else:
            return BigTwoRLWrapper(**kwargs)
    
    @staticmethod
    def detect_wrapper_type(wrapper) -> str:
        """Detect which type of wrapper is being used.
        
        Args:
            wrapper: Wrapper instance
            
        Returns:
            "fixed" or "legacy"
        """
        if isinstance(wrapper, FixedActionBigTwoWrapper):
            return "fixed"
        elif isinstance(wrapper, BigTwoRLWrapper):
            return "legacy"
        else:
            return "unknown"
    
    @staticmethod
    def get_action_space_info(wrapper) -> Dict[str, Any]:
        """Get information about the wrapper's action space.
        
        Args:
            wrapper: Wrapper instance
            
        Returns:
            Dictionary with action space information
        """
        wrapper_type = CompatibilityWrapper.detect_wrapper_type(wrapper)
        
        info = {
            "wrapper_type": wrapper_type,
            "action_space_size": wrapper.action_space.n,
            "is_fixed_action_space": wrapper_type == "fixed"
        }
        
        if wrapper_type == "fixed":
            info["total_possible_actions"] = 1365
            info["uses_action_masking"] = True
        elif wrapper_type == "legacy":
            info["max_dynamic_actions"] = wrapper.action_space.n
            info["uses_action_masking"] = True
            
        return info


class ModelConverter:
    """Converts models between old and new action space formats.
    
    Note: This is a placeholder for future implementation. Converting trained models
    between action spaces is complex and may require retraining.
    """
    
    def __init__(self):
        self.adapter = LegacyActionAdapter()
        
    def convert_legacy_to_fixed_actions(self, legacy_model_path: str, output_path: Optional[str] = None) -> str:
        """Convert a legacy model to work with fixed action space.
        
        Args:
            legacy_model_path: Path to legacy model
            output_path: Path to save converted model (auto-generated if None)
            
        Returns:
            Path to converted model
        """
        raise NotImplementedError(
            "Model conversion between action spaces is complex and not yet implemented. "
            "Consider retraining models with the new fixed action space instead."
        )
    
    def convert_fixed_to_legacy_actions(self, fixed_model_path: str, output_path: Optional[str] = None) -> str:
        """Convert a fixed action space model to work with legacy system.
        
        Args:
            fixed_model_path: Path to fixed action space model
            output_path: Path to save converted model
            
        Returns:
            Path to converted model
        """
        raise NotImplementedError(
            "Model conversion between action spaces is complex and not yet implemented. "
            "Fixed action space models cannot easily be converted to legacy format."
        )
    
    def estimate_conversion_feasibility(self, model_path: str) -> Dict[str, Any]:
        """Estimate how feasible it would be to convert a model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with conversion feasibility information
        """
        return {
            "feasible": False,
            "reason": "Model conversion not implemented",
            "recommendation": "Retrain model with desired action space",
            "alternative": "Use adapter layers for inference-only conversion"
        }


class InferenceAdapter:
    """Adapter for using legacy models with fixed action space or vice versa.
    
    This adapter allows using a trained model with a different action space
    by translating actions at inference time. This is useful for evaluation
    and comparison purposes.
    """
    
    def __init__(self, model, source_format: str, target_format: str):
        """Initialize inference adapter.
        
        Args:
            model: Trained model instance
            source_format: Format model was trained on ("fixed" or "legacy") 
            target_format: Format to adapt to ("fixed" or "legacy")
        """
        self.model = model
        self.source_format = source_format
        self.target_format = target_format
        self.adapter = LegacyActionAdapter()
        
        if source_format == target_format:
            raise ValueError("Source and target formats are the same - no adaptation needed")
    
    def predict(self, observation: np.ndarray, legal_moves: List[np.ndarray], player_hand: np.ndarray) -> int:
        """Make prediction with action space adaptation.
        
        Args:
            observation: Game observation
            legal_moves: Legal moves from game engine
            player_hand: Player's hand
            
        Returns:
            Action in target format
        """
        if self.source_format == "legacy" and self.target_format == "fixed":
            # Model trained on legacy format, need fixed format output
            
            # Get model prediction (legacy action index)
            legacy_action = self._get_model_prediction(observation, len(legal_moves) + 1)  # +1 for pass
            
            # Convert to fixed action ID
            return self.adapter.convert_old_action_to_new(legacy_action, legal_moves, player_hand)
            
        elif self.source_format == "fixed" and self.target_format == "legacy":
            # Model trained on fixed format, need legacy format output
            
            # Get model prediction (fixed action ID)
            fixed_action = self._get_model_prediction(observation, 1365)
            
            # Convert to legacy action index
            return self.adapter.convert_new_action_to_old(fixed_action, legal_moves, player_hand)
        
        else:
            raise ValueError(f"Unsupported format conversion: {self.source_format} -> {self.target_format}")
    
    def _get_model_prediction(self, observation: np.ndarray, action_space_size: int) -> int:
        """Get prediction from the underlying model.
        
        Args:
            observation: Game observation
            action_space_size: Size of action space model expects
            
        Returns:
            Action index/ID from model
        """
        # This is a placeholder - actual implementation would depend on model type
        # For stable-baselines3 models:
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        except Exception:
            # Fallback for different model interfaces
            if hasattr(self.model, 'predict'):
                return int(self.model.predict(observation))
            else:
                raise NotImplementedError("Model prediction interface not recognized")


# Convenience functions for backward compatibility
def create_compatible_wrapper(use_fixed_actions: bool = True, **kwargs):
    """Create wrapper compatible with both action systems."""
    return CompatibilityWrapper.create_wrapper(use_fixed_actions, **kwargs)


def adapt_legacy_agent_for_fixed_actions(legacy_agent, adapter: Optional[InferenceAdapter] = None):
    """Adapt a legacy agent to work with fixed action space."""
    if adapter is None:
        # Create default adapter
        adapter = InferenceAdapter(legacy_agent, "legacy", "fixed")
    return adapter


def get_migration_status(wrapper) -> Dict[str, Any]:
    """Get migration status information for a wrapper."""
    wrapper_info = CompatibilityWrapper.get_action_space_info(wrapper)
    
    status = {
        "current_system": wrapper_info["wrapper_type"],
        "is_migrated": wrapper_info["is_fixed_action_space"],
        "action_space_size": wrapper_info["action_space_size"],
        "migration_required": not wrapper_info["is_fixed_action_space"],
    }
    
    if status["migration_required"]:
        status["migration_steps"] = [
            "1. Update code to use FixedActionBigTwoWrapper",
            "2. Retrain models with fixed action space",
            "3. Update evaluation and tournament code",
            "4. Validate performance against legacy system"
        ]
    else:
        status["migration_steps"] = ["Migration complete - using fixed action space"]
        
    return status