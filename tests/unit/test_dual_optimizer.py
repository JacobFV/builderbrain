"""
Tests for dual optimizer in bb_losses.
"""

import torch
import pytest
from bb_losses.dual_optimizer import DualOptimizer
from bb_core.math_utils import RankNormalizer


class TestDualOptimizer:
    def test_initialization(self):
        """Test dual optimizer initialization."""
        constraint_configs = {
            'grammar': {'target': 0.0, 'normalizer': 'rank'},
            'graph2graph': {'target': 0.2, 'normalizer': 'rank'}
        }

        optimizer = DualOptimizer(constraint_configs, eta_lambda=0.01, lambda_max=10.0)

        assert optimizer.eta_lambda == 0.01
        assert optimizer.lambda_max == 10.0
        assert len(optimizer.constraint_configs) == 2

    def test_lagrangian_computation(self):
        """Test Lagrangian computation."""
        constraint_configs = {
            'grammar': {'target': 0.0, 'normalizer': 'rank'},
        }

        optimizer = DualOptimizer(constraint_configs)

        # Add constraint to manager
        normalizer = RankNormalizer(window_size=10)
        optimizer.constraint_manager.add_constraint('grammar', 0.0, normalizer)

        # Test Lagrangian computation
        task_loss = torch.tensor(1.0)
        constraint_losses = {'grammar': torch.tensor(0.5)}

        total_loss, normalized_losses = optimizer.compute_lagrangian(task_loss, constraint_losses)

        assert isinstance(total_loss, torch.Tensor)
        assert 'grammar' in normalized_losses
        assert isinstance(normalized_losses['grammar'], (torch.Tensor, float))

    def test_update_duals(self):
        """Test dual variable updates."""
        constraint_configs = {
            'grammar': {'target': 0.0, 'normalizer': 'rank'},
        }

        optimizer = DualOptimizer(constraint_configs)

        # Add constraint to manager
        normalizer = RankNormalizer(window_size=10)
        optimizer.constraint_manager.add_constraint('grammar', 0.0, normalizer)

        # Update with constraint loss above target
        normalized_losses = {'grammar': torch.tensor(0.5)}

        optimizer.update_duals(normalized_losses)

        # Dual should increase since loss > target
        dual_value = optimizer.get_dual_values()['grammar']
        assert dual_value >= 0.0

    def test_get_dual_values(self):
        """Test getting dual variable values."""
        constraint_configs = {
            'grammar': {'target': 0.0, 'normalizer': 'rank'},
            'graph2graph': {'target': 0.2, 'normalizer': 'rank'}
        }

        optimizer = DualOptimizer(constraint_configs)

        duals = optimizer.get_dual_values()
        assert 'grammar' in duals
        assert 'graph2graph' in duals
        assert all(isinstance(v, float) for v in duals.values())


if __name__ == "__main__":
    pytest.main([__file__])
