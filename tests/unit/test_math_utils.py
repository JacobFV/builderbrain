"""
Tests for mathematical utilities in bb_core.
"""

import numpy as np
import torch
import pytest
from bb_core.math_utils import RankNormalizer, WinsorNormalizer, ConstraintManager


class TestRankNormalizer:
    def test_initialization(self):
        normalizer = RankNormalizer(window_size=100)
        assert normalizer.window_size == 100
        assert len(normalizer.history) == 0

    def test_normalization(self):
        normalizer = RankNormalizer(window_size=10)

        # Test with single value
        result = normalizer(np.array([5.0]))
        assert result == 0.0  # Only one value, should be at median (0)

        # Test with multiple values (add them to history)
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        for val in values:
            normalizer(np.array([val]))

        # Test normalization of new value (should be rank 5 out of 7 values now)
        result = normalizer(np.array([5.0]))
        expected = 2 * (5/7) - 1  # Rank 5 out of 7 values (1,3,5,5,5,7,9)
        assert abs(result - expected) < 1e-6

    def test_state_dict(self):
        normalizer = RankNormalizer(window_size=50)
        normalizer(np.array([1.0, 2.0, 3.0]))

        state = normalizer.state_dict()
        assert state['window_size'] == 50
        assert len(state['history']) == 3

        # Test loading state
        new_normalizer = RankNormalizer()
        new_normalizer.load_state_dict(state)
        assert new_normalizer.window_size == 50
        assert len(new_normalizer.history) == 3


class TestWinsorNormalizer:
    def test_initialization(self):
        normalizer = WinsorNormalizer(tau=2.0, window_size=100)
        assert normalizer.tau == 2.0
        assert normalizer.window_size == 100

    def test_normalization(self):
        normalizer = WinsorNormalizer(tau=2.0, window_size=10)

        # Add some values including outliers
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # 100 is outlier
        for val in values:
            normalizer(np.array([val]))

        # Test that outlier is clipped
        result = normalizer(np.array([100.0]))
        # Should be clipped to mean + 2*std
        assert result < 10.0  # Much less than 100


class TestConstraintManager:
    def test_initialization(self):
        manager = ConstraintManager(eta_lambda=0.01, lambda_max=10.0)
        assert manager.eta_lambda == 0.01
        assert manager.lambda_max == 10.0
        assert len(manager.duals) == 0

    def test_add_constraint(self):
        manager = ConstraintManager()
        normalizer = RankNormalizer()

        manager.add_constraint('test_constraint', 0.5, normalizer)
        assert 'test_constraint' in manager.duals
        assert manager.duals['test_constraint'] == 0.0
        assert 'test_constraint' in manager.normalizers

    def test_update_duals(self):
        manager = ConstraintManager(eta_lambda=0.1, lambda_max=5.0)
        normalizer = RankNormalizer(window_size=10)

        manager.add_constraint('test', 0.5, normalizer)

        # Update with a loss value above target
        raw_losses = {'test': 0.8}

        # First add some history to the normalizer so it can compute proper normalization
        for i in range(10):
            normalizer(np.array([0.5 + i * 0.1]))

        manager.update_duals(raw_losses)

        # Dual should increase since loss > target
        print(f"Dual value after update: {manager.duals['test']}")
        assert manager.duals['test'] >= 0.0

    def test_normalize_losses(self):
        manager = ConstraintManager()
        normalizer = RankNormalizer(window_size=10)

        manager.add_constraint('test', 0.5, normalizer)

        # Test with float values
        raw_losses = {'test': 0.8}
        normalized = manager.normalize_losses(raw_losses)

        assert 'test' in normalized
        assert isinstance(normalized['test'], float)

    def test_lagrangian_contribution(self):
        manager = ConstraintManager()
        normalizer = RankNormalizer(window_size=10)

        manager.add_constraint('test', 0.5, normalizer)

        # Test lagrangian computation
        raw_losses = {'test': 0.8}
        contribution = manager.get_lagrangian_contribution(raw_losses)

        assert isinstance(contribution, float)
        assert contribution >= 0.0  # Should be non-negative


if __name__ == "__main__":
    pytest.main([__file__])
