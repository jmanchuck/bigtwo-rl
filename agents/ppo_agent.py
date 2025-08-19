"""PPO agent wrapper for stable-baselines3 models."""

import numpy as np
from stable_baselines3 import PPO
from .base_agent import BaseAgent

class PPOAgent(BaseAgent):
    """PPO agent wrapper for trained models."""
    
    def __init__(self, model_path=None, model=None, name="PPO"):
        super().__init__(name)
        if model_path:
            self.model = PPO.load(model_path)
        elif model:
            self.model = model
        else:
            raise ValueError("Must provide either model_path or model")
        
        self.deterministic = True
    
    def get_action(self, observation, action_mask=None):
        """Get action from PPO model."""
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(action)
    
    def reset(self):
        """Nothing to reset for PPO agent (stateless)."""
        pass
    
    def set_deterministic(self, deterministic: bool):
        """Set whether to use deterministic policy."""
        self.deterministic = deterministic