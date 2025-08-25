"""Configuration system for Big Two RL with feature flags and migration support.

This module provides configuration management for transitioning between legacy
and fixed action space systems, with environment variable controls and factory functions.
"""

import os
import warnings
from typing import Union, Type, Dict, Any, Optional, List
from enum import Enum

# Import classes for type hints and factory functions
from .core.rl_wrapper import BigTwoRLWrapper
from .core.fixed_action_wrapper import FixedActionBigTwoWrapper
from .training.trainer import Trainer
from .training.fixed_action_trainer import FixedActionTrainer
from .evaluation.evaluator import Evaluator
from .evaluation.fixed_action_evaluator import FixedActionEvaluator


class ActionSpaceType(Enum):
    """Action space types available in the system."""
    LEGACY = "legacy"
    FIXED = "fixed"


class MigrationPhase(Enum):
    """Migration phases for transitioning between action spaces."""
    LEGACY_ONLY = "legacy_only"
    COEXISTENCE = "coexistence"
    FIXED_DEFAULT = "fixed_default"
    FIXED_ONLY = "fixed_only"


# Configuration from environment variables
USE_FIXED_ACTION_SPACE = os.getenv('BIGTWO_USE_FIXED_ACTIONS', 'true').lower() == 'true'
ALLOW_LEGACY_FALLBACK = os.getenv('BIGTWO_ALLOW_LEGACY', 'true').lower() == 'true'
MIGRATION_PHASE = os.getenv('BIGTWO_MIGRATION_PHASE', 'fixed_default').lower()
ENABLE_MIGRATION_WARNINGS = os.getenv('BIGTWO_MIGRATION_WARNINGS', 'true').lower() == 'true'
FORCE_COMPATIBILITY_MODE = os.getenv('BIGTWO_FORCE_COMPATIBILITY', 'false').lower() == 'true'

# Validate migration phase
valid_phases = [phase.value for phase in MigrationPhase]
if MIGRATION_PHASE not in valid_phases:
    print(f"Warning: Invalid migration phase '{MIGRATION_PHASE}'. Using 'fixed_default'.")
    MIGRATION_PHASE = 'fixed_default'


class BigTwoConfig:
    """Central configuration class for Big Two RL system."""
    
    def __init__(self):
        self.action_space_type = ActionSpaceType.FIXED if USE_FIXED_ACTION_SPACE else ActionSpaceType.LEGACY
        self.migration_phase = MigrationPhase(MIGRATION_PHASE)
        self.allow_legacy_fallback = ALLOW_LEGACY_FALLBACK
        self.enable_warnings = ENABLE_MIGRATION_WARNINGS
        self.force_compatibility = FORCE_COMPATIBILITY_MODE
    
    def get_wrapper_class(self) -> Type[Union[BigTwoRLWrapper, FixedActionBigTwoWrapper]]:
        """Get appropriate wrapper class based on configuration."""
        return get_wrapper_class()
    
    def get_trainer_class(self) -> Type[Union[Trainer, FixedActionTrainer]]:
        """Get appropriate trainer class based on configuration."""
        return get_trainer_class()
    
    def get_evaluator_class(self) -> Type[Union[Evaluator, FixedActionEvaluator]]:
        """Get appropriate evaluator class based on configuration."""
        return get_evaluator_class()
    
    def is_using_fixed_actions(self) -> bool:
        """Check if system is configured to use fixed action space."""
        return self.action_space_type == ActionSpaceType.FIXED
    
    def get_status(self) -> Dict[str, Any]:
        """Get current configuration status."""
        return {
            'action_space_type': self.action_space_type.value,
            'migration_phase': self.migration_phase.value,
            'allow_legacy_fallback': self.allow_legacy_fallback,
            'enable_warnings': self.enable_warnings,
            'force_compatibility': self.force_compatibility,
            'environment_variables': {
                'BIGTWO_USE_FIXED_ACTIONS': USE_FIXED_ACTION_SPACE,
                'BIGTWO_ALLOW_LEGACY': ALLOW_LEGACY_FALLBACK,
                'BIGTWO_MIGRATION_PHASE': MIGRATION_PHASE,
                'BIGTWO_MIGRATION_WARNINGS': ENABLE_MIGRATION_WARNINGS,
                'BIGTWO_FORCE_COMPATIBILITY': FORCE_COMPATIBILITY_MODE,
            }
        }


# Global configuration instance
config = BigTwoConfig()


def get_trainer_class() -> Type[Union[Trainer, FixedActionTrainer]]:
    """Get trainer class based on configuration.
    
    Returns:
        Appropriate trainer class
    """
    if config.migration_phase == MigrationPhase.LEGACY_ONLY:
        return Trainer
    elif config.migration_phase == MigrationPhase.FIXED_ONLY:
        return FixedActionTrainer
    elif USE_FIXED_ACTION_SPACE:
        if config.enable_warnings and config.migration_phase == MigrationPhase.COEXISTENCE:
            print("ℹ️  Using FixedActionTrainer (fixed action space enabled)")
        return FixedActionTrainer
    elif ALLOW_LEGACY_FALLBACK:
        if config.enable_warnings:
            warnings.warn(
                "Using legacy Trainer. Fixed action space is recommended for better performance. "
                "Set BIGTWO_USE_FIXED_ACTIONS=true to enable.",
                FutureWarning,
                stacklevel=2
            )
        return Trainer
    else:
        raise RuntimeError("Legacy system disabled, but fixed action space not enabled")


def get_wrapper_class() -> Type[Union[BigTwoRLWrapper, FixedActionBigTwoWrapper]]:
    """Get wrapper class based on configuration.
    
    Returns:
        Appropriate wrapper class
    """
    if config.migration_phase == MigrationPhase.LEGACY_ONLY:
        return BigTwoRLWrapper
    elif config.migration_phase == MigrationPhase.FIXED_ONLY:
        return FixedActionBigTwoWrapper
    elif USE_FIXED_ACTION_SPACE:
        if config.enable_warnings and config.migration_phase == MigrationPhase.COEXISTENCE:
            print("ℹ️  Using FixedActionBigTwoWrapper (fixed action space enabled)")
        return FixedActionBigTwoWrapper
    elif ALLOW_LEGACY_FALLBACK:
        if config.enable_warnings:
            warnings.warn(
                "Using legacy BigTwoRLWrapper. Fixed action space is recommended for better performance. "
                "Set BIGTWO_USE_FIXED_ACTIONS=true to enable.",
                FutureWarning,
                stacklevel=2
            )
        return BigTwoRLWrapper
    else:
        raise RuntimeError("Legacy system disabled, but fixed action space not enabled")


def get_evaluator_class() -> Type[Union[Evaluator, FixedActionEvaluator]]:
    """Get evaluator class based on configuration.
    
    Returns:
        Appropriate evaluator class
    """
    if config.migration_phase == MigrationPhase.LEGACY_ONLY:
        return Evaluator
    elif config.migration_phase == MigrationPhase.FIXED_ONLY:
        return FixedActionEvaluator
    elif USE_FIXED_ACTION_SPACE:
        if config.enable_warnings and config.migration_phase == MigrationPhase.COEXISTENCE:
            print("ℹ️  Using FixedActionEvaluator (fixed action space enabled)")
        return FixedActionEvaluator
    elif ALLOW_LEGACY_FALLBACK:
        if config.enable_warnings:
            warnings.warn(
                "Using legacy Evaluator. Fixed action space is recommended for better performance. "
                "Set BIGTWO_USE_FIXED_ACTIONS=true to enable.",
                FutureWarning,
                stacklevel=2
            )
        return Evaluator
    else:
        raise RuntimeError("Legacy system disabled, but fixed action space not enabled")


def create_trainer(**kwargs):
    """Factory function for trainer creation.
    
    Args:
        **kwargs: Arguments passed to trainer constructor
        
    Returns:
        Configured trainer instance
    """
    trainer_class = get_trainer_class()
    return trainer_class(**kwargs)


def create_wrapper(**kwargs):
    """Factory function for wrapper creation.
    
    Args:
        **kwargs: Arguments passed to wrapper constructor
        
    Returns:
        Configured wrapper instance
    """
    wrapper_class = get_wrapper_class()
    return wrapper_class(**kwargs)


def create_evaluator(**kwargs):
    """Factory function for evaluator creation.
    
    Args:
        **kwargs: Arguments passed to evaluator constructor
        
    Returns:
        Configured evaluator instance
    """
    evaluator_class = get_evaluator_class()
    return evaluator_class(**kwargs)


# Migration and status functions
def is_using_fixed_actions() -> bool:
    """Check if system is using fixed action space.
    
    Returns:
        True if using fixed action space
    """
    return USE_FIXED_ACTION_SPACE


def migration_status() -> str:
    """Get current migration status.
    
    Returns:
        Migration status string
    """
    if config.migration_phase == MigrationPhase.FIXED_ONLY:
        return "FIXED_ACTION_SPACE_ONLY"
    elif config.migration_phase == MigrationPhase.LEGACY_ONLY:
        return "LEGACY_SYSTEM_ONLY"
    elif USE_FIXED_ACTION_SPACE:
        return "FIXED_ACTION_SPACE_ACTIVE"
    elif ALLOW_LEGACY_FALLBACK:
        return "LEGACY_FALLBACK_AVAILABLE" 
    else:
        return "CONFIGURATION_ERROR"


def get_configuration_info() -> Dict[str, Any]:
    """Get comprehensive configuration information.
    
    Returns:
        Dictionary with configuration details
    """
    return {
        'current_configuration': config.get_status(),
        'migration_status': migration_status(),
        'recommended_actions': get_migration_recommendations(),
        'class_mappings': {
            'trainer': get_trainer_class().__name__,
            'wrapper': get_wrapper_class().__name__,
            'evaluator': get_evaluator_class().__name__,
        }
    }


def get_migration_recommendations() -> List[str]:
    """Get migration recommendations based on current configuration.
    
    Returns:
        List of recommended actions
    """
    recommendations = []
    
    if config.migration_phase == MigrationPhase.LEGACY_ONLY:
        recommendations.extend([
            "Consider migrating to fixed action space for better performance",
            "Set BIGTWO_MIGRATION_PHASE=coexistence to enable both systems",
            "Test fixed action space with BIGTWO_USE_FIXED_ACTIONS=true"
        ])
    elif config.migration_phase == MigrationPhase.COEXISTENCE:
        if not USE_FIXED_ACTION_SPACE:
            recommendations.extend([
                "Enable fixed action space with BIGTWO_USE_FIXED_ACTIONS=true",
                "Compare performance between legacy and fixed action systems",
                "Consider setting BIGTWO_MIGRATION_PHASE=fixed_default"
            ])
        else:
            recommendations.extend([
                "Migration in progress - using fixed action space",
                "Test thoroughly before setting BIGTWO_MIGRATION_PHASE=fixed_only",
                "Consider disabling legacy fallback once migration is complete"
            ])
    elif config.migration_phase == MigrationPhase.FIXED_DEFAULT:
        recommendations.extend([
            "Migration nearly complete - using fixed action space by default",
            "Monitor performance and stability",
            "Set BIGTWO_MIGRATION_PHASE=fixed_only when ready to remove legacy support"
        ])
    elif config.migration_phase == MigrationPhase.FIXED_ONLY:
        recommendations.extend([
            "Migration complete - using fixed action space exclusively",
            "Legacy system is disabled",
            "All new development should use fixed action space APIs"
        ])
    
    return recommendations


def print_configuration_status():
    """Print current configuration status to console."""
    info = get_configuration_info()
    
    print("🔧 Big Two RL Configuration Status")
    print("=" * 50)
    
    current = info['current_configuration']
    print(f"Action Space: {current['action_space_type'].upper()}")
    print(f"Migration Phase: {current['migration_phase'].upper()}")
    print(f"Migration Status: {info['migration_status']}")
    
    print(f"\nClass Mappings:")
    for component, class_name in info['class_mappings'].items():
        print(f"  {component.title()}: {class_name}")
    
    if info['recommended_actions']:
        print(f"\n📋 Recommendations:")
        for i, recommendation in enumerate(info['recommended_actions'], 1):
            print(f"  {i}. {recommendation}")
    
    print("=" * 50)


def validate_configuration():
    """Validate current configuration and report any issues."""
    issues = []
    warnings_list = []
    
    # Check for conflicting environment variables
    if not USE_FIXED_ACTION_SPACE and not ALLOW_LEGACY_FALLBACK:
        issues.append("Both fixed actions and legacy fallback are disabled")
    
    # Check migration phase consistency
    if config.migration_phase == MigrationPhase.FIXED_ONLY and ALLOW_LEGACY_FALLBACK:
        warnings_list.append("Migration phase is 'fixed_only' but legacy fallback is enabled")
    
    if config.migration_phase == MigrationPhase.LEGACY_ONLY and USE_FIXED_ACTION_SPACE:
        warnings_list.append("Migration phase is 'legacy_only' but fixed actions are enabled")
    
    # Report issues
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings_list:
        print("⚠️  Configuration Warnings:")
        for warning in warnings_list:
            print(f"  - {warning}")
    
    if not issues and not warnings_list:
        print("✅ Configuration is valid")
    
    return len(issues) == 0


# Environment variable documentation
ENVIRONMENT_VARIABLES = {
    'BIGTWO_USE_FIXED_ACTIONS': {
        'description': 'Enable fixed 1365-action space (true/false)',
        'default': 'true',
        'current': str(USE_FIXED_ACTION_SPACE).lower()
    },
    'BIGTWO_ALLOW_LEGACY': {
        'description': 'Allow fallback to legacy action space (true/false)',
        'default': 'true',
        'current': str(ALLOW_LEGACY_FALLBACK).lower()
    },
    'BIGTWO_MIGRATION_PHASE': {
        'description': 'Migration phase (legacy_only/coexistence/fixed_default/fixed_only)',
        'default': 'fixed_default',
        'current': MIGRATION_PHASE
    },
    'BIGTWO_MIGRATION_WARNINGS': {
        'description': 'Enable migration warnings (true/false)',
        'default': 'true',
        'current': str(ENABLE_MIGRATION_WARNINGS).lower()
    },
    'BIGTWO_FORCE_COMPATIBILITY': {
        'description': 'Force compatibility mode (true/false)',
        'default': 'false',
        'current': str(FORCE_COMPATIBILITY_MODE).lower()
    }
}


def print_environment_variables():
    """Print documentation for environment variables."""
    print("🌍 Environment Variables")
    print("=" * 50)
    
    for var_name, info in ENVIRONMENT_VARIABLES.items():
        print(f"{var_name}:")
        print(f"  Description: {info['description']}")
        print(f"  Default: {info['default']}")
        print(f"  Current: {info['current']}")
        print()


if __name__ == "__main__":
    # When run directly, show configuration status
    print_configuration_status()
    print()
    validate_configuration()
    print()
    print_environment_variables()