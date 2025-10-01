"""
Configuration management for BuilderBrain.

Loads and manages configurations for different model scales and use cases.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_config_for_scale(scale: str) -> Dict[str, Any]:
    """Get configuration for a specific scale."""
    config_paths = {
        'tiny': 'configs/tiny.yaml',
        'small': 'configs/small.yaml',
        'medium': 'configs/medium.yaml',
        'large': 'configs/large.yaml',
        'production': 'configs/production.yaml'
    }

    if scale not in config_paths:
        raise ValueError(f"Unknown scale: {scale}. Available: {list(config_paths.keys())}")

    return load_config(config_paths[scale])


def create_config_tiny() -> Dict[str, Any]:
    """Create tiny configuration for testing."""
    return {
        'model': {
            'type': 'tiny',
            'name': 'tiny',
            'hidden_size': 64,
            'num_layers': 2,
            'num_programs': 8,
            'alpha_cap': 0.05
        },
        'constraints': {
            'grammar': {
                'enabled': True,
                'target': 0.0,
                'normalizer': 'rank'
            },
            'graph2graph': {
                'enabled': True,
                'target': 0.2,
                'normalizer': 'rank'
            }
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-3,
            'eta_lambda': 1e-2,
            'lambda_max': 10.0,
            'num_epochs': 10
        }
    }


def create_config_small() -> Dict[str, Any]:
    """Create small configuration for development."""
    return {
        'model': {
            'type': 'gpt2',
            'name': 'gpt2',
            'hidden_size': 768,
            'num_layers': 4,
            'num_programs': 16,
            'alpha_cap': 0.1
        },
        'constraints': {
            'grammar': {
                'enabled': True,
                'target': 0.0,
                'normalizer': 'rank'
            },
            'graph2graph': {
                'enabled': True,
                'target': 0.2,
                'normalizer': 'rank'
            },
            'buildability': {
                'enabled': True,
                'target': 0.0,
                'normalizer': 'winsor'
            }
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 5e-4,
            'eta_lambda': 1e-2,
            'lambda_max': 20.0,
            'num_epochs': 50
        }
    }


def create_config_production() -> Dict[str, Any]:
    """Create production configuration."""
    return {
        'model': {
            'type': 'gpt_neo',
            'name': 'EleutherAI/gpt-neo-2.7B',
            'hidden_size': 2560,
            'num_layers': 8,
            'num_programs': 32,
            'alpha_cap': 0.15
        },
        'constraints': {
            'grammar': {
                'enabled': True,
                'target': 0.0,
                'normalizer': 'rank'
            },
            'graph2graph': {
                'enabled': True,
                'target': 0.15,
                'normalizer': 'rank'
            },
            'buildability': {
                'enabled': True,
                'target': 0.0,
                'normalizer': 'winsor'
            },
            'reuse': {
                'enabled': True,
                'target': 0.3,
                'normalizer': 'rank'
            },
            'calibration': {
                'enabled': True,
                'target': 0.05,
                'normalizer': 'rank'
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'eta_lambda': 5e-3,
            'lambda_max': 50.0,
            'num_epochs': 100,
            'gradient_checkpointing': True,
            'mixed_precision': True
        }
    }
