"""PPO agent for fixed 1,365-action space.

This agent works with models trained using the fixed action space and
provides proper action masking during inference.
"""

import torch
import numpy as np
from typing import Optional, Any, Dict
from pathlib import Path

from .base_agent import BaseAgent
from ..training.multi_player_ppo import MultiPlayerPPO
from ..core.action_system import BigTwoActionSystem
from ..core.observation_builder import ObservationConfig


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class FixedActionPPOAgent(BaseAgent):
    """PPO agent for fixed 1,365-action space.
    
    This agent loads models trained with the fixed action space and
    handles action masking during inference for optimal play.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        name: str = "FixedActionPPO",
        observation_config: Optional[ObservationConfig] = None,
        deterministic: bool = True,
    ) -> None:
        """Initialize Fixed Action PPO agent.
        
        Args:
            model_path: Path to trained PPO model
            name: Agent name for identification
            observation_config: Observation configuration (for validation)
            deterministic: Whether to use deterministic actions
        """
        super().__init__(name)
        self.model_path = model_path
        self.action_system = BigTwoActionSystem()
        self.deterministic = deterministic
        self.observation_config = observation_config
        self.model = None
        
        # Load model if path provided
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load PPO model from file.
        
        Args:
            model_path: Path to model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = MultiPlayerPPO.load(model_path)
            
            # Validate model expects 1365 actions
            if self.model.action_space.n != 1365:
                raise ValueError(
                    f"Model expects {self.model.action_space.n} actions, "
                    f"but fixed space has 1365"
                )
            
            print(f"✅ Loaded fixed action PPO model: {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def get_action(
        self, 
        observation: np.ndarray, 
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Get action using fixed action space.
        
        Args:
            observation: Game observation vector
            action_mask: 1365-dim boolean mask for legal actions
            
        Returns:
            action_id: Action ID from 0-1364
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not loaded - call _load_model() first")
        
        # Get model prediction (1365 logits)
        try:
            # Handle both single observations and batches
            if observation.ndim == 1:
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            else:
                obs_tensor = torch.FloatTensor(observation)
            
            with torch.no_grad():
                # Get action distribution from policy
                distribution = self.model.policy.get_distribution(obs_tensor)
                
                if action_mask is not None:
                    # Apply action mask to distribution
                    distribution.set_mask(torch.BoolTensor(action_mask).unsqueeze(0))
                
                if self.deterministic:
                    action = distribution.mode()
                else:
                    action = distribution.sample()
                
                return int(action.item())
                
        except Exception as e:
            # Fallback to raw logits approach
            try:
                action, _ = self.model.predict(observation, deterministic=self.deterministic)
                action_id = int(action)
                
                # Apply masking if provided
                if action_mask is not None:
                    if not action_mask[action_id]:
                        # Action is masked, select best valid action
                        valid_actions = np.where(action_mask)[0]
                        if len(valid_actions) > 0:
                            # Get logits and select best valid action
                            with torch.no_grad():
                                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                                logits = self.model.policy.get_distribution(obs_tensor).distribution.logits
                                masked_logits = torch.where(
                                    torch.BoolTensor(action_mask),
                                    logits.squeeze(),
                                    torch.tensor(-float('inf'))
                                )
                                action_id = int(torch.argmax(masked_logits).item())
                        else:
                            raise ValueError("No valid actions available")
                
                return action_id
                
            except Exception as e2:
                raise RuntimeError(f"Failed to get action: {e}, fallback also failed: {e2}")
    
    def get_action_with_info(
        self,
        observation: np.ndarray,
        action_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Get action with additional information.
        
        Args:
            observation: Game observation
            action_mask: Legal action mask
            
        Returns:
            Dictionary with action and additional info
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            with torch.no_grad():
                # Get full policy output
                actions, values, log_probs = self.model.policy(obs_tensor)
                distribution = self.model.policy.get_distribution(obs_tensor)
                
                # Apply masking if provided
                if action_mask is not None:
                    distribution.set_mask(torch.BoolTensor(action_mask).unsqueeze(0))
                    
                    if self.deterministic:
                        action = distribution.mode()
                    else:
                        action = distribution.sample()
                    
                    log_prob = distribution.log_prob(action)
                else:
                    action = actions
                    log_prob = log_probs
                
                # Get action probabilities for analysis
                probs = torch.softmax(distribution.distribution.logits, dim=-1)
                
                info = {
                    'action': int(action.item()),
                    'value': float(values.item()),
                    'log_prob': float(log_prob.item()),
                    'action_probs': probs.squeeze().numpy(),
                    'entropy': float(distribution.entropy().item()),
                    'valid_actions_count': int(action_mask.sum()) if action_mask is not None else 1365,
                }
                
                return info
                
        except Exception as e:
            # Fallback to simple prediction
            action = self.get_action(observation, action_mask)
            return {
                'action': action,
                'value': None,
                'log_prob': None,
                'action_probs': None,
                'entropy': None,
                'valid_actions_count': int(action_mask.sum()) if action_mask is not None else 1365,
                'fallback_used': True,
                'error': str(e)
            }
    
    def reset(self) -> None:
        """Reset agent state (no-op for stateless agent)."""
        pass
    
    def set_deterministic(self, deterministic: bool) -> None:
        """Set whether to use deterministic or stochastic actions.
        
        Args:
            deterministic: If True, always select highest probability action
        """
        self.deterministic = deterministic
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not hasattr(self, 'model') or self.model is None:
            return {'model_loaded': False}
        
        info = {
            'model_loaded': True,
            'model_path': self.model_path,
            'action_space_size': self.model.action_space.n,
            'observation_space_shape': self.model.observation_space.shape,
            'policy_type': type(self.model.policy).__name__,
            'deterministic': self.deterministic,
        }
        
        # Add training info if available
        try:
            if hasattr(self.model, 'num_timesteps'):
                info['training_timesteps'] = self.model.num_timesteps
        except:
            pass
        
        return info
    
    def validate_compatibility(self, observation_space, action_space) -> Dict[str, bool]:
        """Validate agent compatibility with environment.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            
        Returns:
            Dictionary with compatibility checks
        """
        if not hasattr(self, 'model') or self.model is None:
            return {'model_loaded': False}
        
        checks = {
            'model_loaded': True,
            'action_space_compatible': action_space.n == 1365,
            'observation_space_compatible': (
                observation_space.shape == self.model.observation_space.shape
            ),
            'is_fixed_action_space': action_space.n == 1365,
        }
        
        checks['fully_compatible'] = all([
            checks['action_space_compatible'],
            checks['observation_space_compatible'],
            checks['is_fixed_action_space']
        ])
        
        return checks


class FixedActionPPOAgentEnsemble:
    """Ensemble of multiple fixed action PPO agents.
    
    This can be used for more robust decision making by combining
    predictions from multiple trained models.
    """
    
    def __init__(self, model_paths: list, name: str = "PPOEnsemble"):
        """Initialize ensemble of PPO agents.
        
        Args:
            model_paths: List of paths to trained models
            name: Ensemble name
        """
        self.name = name
        self.agents = []
        
        for i, path in enumerate(model_paths):
            agent = FixedActionPPOAgent(
                model_path=path,
                name=f"{name}_Agent_{i}",
                deterministic=True
            )
            self.agents.append(agent)
        
        print(f"✅ Created ensemble with {len(self.agents)} agents")
    
    def get_action(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Get action using ensemble voting.
        
        Args:
            observation: Game observation
            action_mask: Legal action mask
            
        Returns:
            Most voted action ID
        """
        if not self.agents:
            raise ValueError("No agents in ensemble")
        
        # Get predictions from all agents
        actions = []
        for agent in self.agents:
            try:
                action = agent.get_action(observation, action_mask)
                actions.append(action)
            except Exception as e:
                print(f"Warning: Agent {agent.name} failed: {e}")
        
        if not actions:
            raise RuntimeError("All agents failed to predict")
        
        # Return most common action (simple voting)
        unique_actions, counts = np.unique(actions, return_counts=True)
        return int(unique_actions[np.argmax(counts)])
    
    def get_consensus_info(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get detailed consensus information from ensemble.
        
        Args:
            observation: Game observation
            action_mask: Legal action mask
            
        Returns:
            Dictionary with ensemble consensus details
        """
        predictions = []
        
        for agent in self.agents:
            try:
                info = agent.get_action_with_info(observation, action_mask)
                predictions.append(info)
            except Exception as e:
                print(f"Warning: Agent {agent.name} failed: {e}")
        
        if not predictions:
            raise RuntimeError("All agents failed to predict")
        
        # Analyze consensus
        actions = [p['action'] for p in predictions]
        unique_actions, counts = np.unique(actions, return_counts=True)
        consensus_action = int(unique_actions[np.argmax(counts)])
        consensus_strength = np.max(counts) / len(actions)
        
        return {
            'consensus_action': consensus_action,
            'consensus_strength': consensus_strength,
            'individual_actions': actions,
            'num_agents': len(self.agents),
            'action_distribution': dict(zip(unique_actions.tolist(), counts.tolist())),
            'all_predictions': predictions
        }