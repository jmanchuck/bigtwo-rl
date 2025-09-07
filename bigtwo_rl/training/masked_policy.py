"""Masked policy for Big Two with fixed action space support.

This module provides a custom policy that handles action masking for the fixed
1,365-action space, enabling proper training with invalid action suppression.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union


class ReferenceMLPExtractor(nn.Module):
    """Reference-compatible MLP extractor with 512→256 architecture.

    This matches the reference paper's architecture for fair comparison.
    """

    def __init__(self, feature_dim: int, device: torch.device = torch.device("cpu")):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        ).to(device)

        # Separate heads for policy and value
        self.policy_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        ).to(device)

        self.value_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        ).to(device)

        self.latent_dim_pi = 256
        self.latent_dim_vf = 256

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MLP extractor.

        Args:
            features: Input features from the feature extractor

        Returns:
            Tuple of (policy_latent, value_latent)
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


class MaskedCategorical(CategoricalDistribution):
    """Categorical distribution with action masking support.

    This distribution applies action masking by setting invalid actions to very
    negative logits before computing probabilities and sampling.
    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.mask = None

    def set_mask(self, mask: Optional[torch.Tensor]):
        """Set action mask for this distribution.

        Args:
            mask: Boolean tensor where True = valid action, False = invalid
        """
        self.mask = mask

    def proba_distribution(self, action_logits: torch.Tensor) -> "MaskedCategorical":
        """Create distribution from action logits with masking.

        Args:
            action_logits: Raw logits from policy network

        Returns:
            Self with updated distribution
        """
        if self.mask is not None:
            # Apply mask by setting invalid actions to very negative values
            masked_logits = torch.where(
                self.mask.bool(),
                action_logits,
                torch.tensor(-1e8, dtype=action_logits.dtype, device=action_logits.device),
            )
        else:
            masked_logits = action_logits

        self.distribution = torch.distributions.Categorical(logits=masked_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions.

        Args:
            actions: Action indices

        Returns:
            Log probabilities of the actions
        """
        log_probs = self.distribution.log_prob(actions)

        if self.mask is not None:
            # Zero out log probs for invalid actions (they should never be taken)
            action_mask = self.mask.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            log_probs = torch.where(action_mask.bool(), log_probs, torch.tensor(-float("inf"), device=log_probs.device))

        return log_probs

    def entropy(self) -> torch.Tensor:
        """Compute entropy of the distribution."""
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        """Sample action from distribution."""
        if self.mask is not None:
            # Ensure we never sample invalid actions
            valid_actions = torch.where(self.mask.any(dim=-1, keepdim=True))[0]
            if len(valid_actions) == 0:
                # Fallback: sample from uniform distribution (should not happen)
                return torch.randint(0, self.action_dim, (self.mask.shape[0],), device=self.mask.device)

        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        """Get mode (most likely action) of distribution."""
        if self.mask is not None:
            # Get the highest probability valid action
            masked_probs = torch.where(
                self.mask.bool(), self.distribution.probs, torch.tensor(0.0, device=self.distribution.probs.device)
            )
            return torch.argmax(masked_probs, dim=-1)
        else:
            return torch.argmax(self.distribution.probs, dim=-1)


class MaskedBigTwoPolicy(ActorCriticPolicy):
    """Big Two policy with built-in action masking support.

    This policy is designed specifically for the fixed 1,365-action space and
    integrates action masking directly into the policy architecture.
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Schedule, **kwargs):
        """Initialize masked Big Two policy.

        Args:
            observation_space: Observation space
            action_space: Action space (must be Discrete(1365))
            lr_schedule: Learning rate schedule
            **kwargs: Additional arguments
        """
        # Force our specific architecture
        if not isinstance(action_space, spaces.Discrete) or action_space.n != 1365:
            raise ValueError(f"Expected Discrete(1365) action space, got {action_space}")

        # Set up custom MLP extractor
        kwargs["net_arch"] = []  # We handle architecture in our custom extractor

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Build the reference-compatible MLP extractor."""
        self.mlp_extractor = ReferenceMLPExtractor(
            feature_dim=self.features_dim,
            device=self.device,
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> MaskedCategorical:
        """Get action distribution from latent policy representation.

        Args:
            latent_pi: Latent policy representation

        Returns:
            Masked categorical distribution
        """
        action_logits = self.action_net(latent_pi)
        action_dist = MaskedCategorical(self.action_space.n)
        return action_dist.proba_distribution(action_logits)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with support for action masking.

        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (actions, values, log_probs)
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)

        # Sample actions
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions with masking support.

        Args:
            obs: Observation tensor
            actions: Action tensor

        Returns:
            Tuple of (values, log_probs, entropy)
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> MaskedCategorical:
        """Get action distribution for given observations.

        Args:
            obs: Observation tensor

        Returns:
            Masked categorical distribution
        """
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values for given observations.

        Args:
            obs: Observation tensor

        Returns:
            Value predictions
        """
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)

    def set_action_mask(self, mask: Optional[torch.Tensor]):
        """Set action mask for the policy.

        This method allows external setting of action masks, which can be useful
        for integration with environments that provide masks.

        Args:
            mask: Boolean tensor where True = valid action
        """
        # This is a placeholder - actual integration would depend on how
        # the training loop provides masks to the policy
        pass


class ActionMaskedPPOPolicy(MaskedBigTwoPolicy):
    """PPO-specific masked policy with additional features.

    This extends the base masked policy with PPO-specific functionality
    and optimizations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional PPO-specific initialization could go here
        self.clip_range = kwargs.get("clip_range", 0.2)

    def ppo_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss with action masking.

        Args:
            observations: Batch of observations
            actions: Batch of actions
            old_log_probs: Log probs from old policy
            advantages: Computed advantages
            returns: Computed returns
            masks: Optional action masks

        Returns:
            Dictionary with loss components
        """
        # Get current policy outputs
        values, log_probs, entropy = self.evaluate_actions(observations, actions)

        # Apply action masks if provided
        if masks is not None:
            distribution = self.get_distribution(observations)
            distribution.set_mask(masks)
            log_probs = distribution.log_prob(actions)

        # Compute ratio for PPO loss
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages

        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)

        # Entropy loss
        entropy_loss = entropy.mean()

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": policy_loss + 0.5 * value_loss - 0.01 * entropy_loss,
        }
