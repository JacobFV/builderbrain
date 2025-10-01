"""
Normalization strategies for loss functions and metrics.

Provides robust normalization that handles outliers and distribution shifts.
"""

from typing import Dict, Any, List, Union
import numpy as np
from .math_utils import Normalizer, RankNormalizer, WinsorNormalizer


class NormalizationManager:
    """Manages normalization strategies for different losses."""

    def __init__(self):
        self.normalizers: Dict[str, Normalizer] = {}

    def add_loss(self, name: str, normalizer_type: str = "rank", **kwargs):
        """Add normalization for a specific loss."""
        if normalizer_type == "rank":
            self.normalizers[name] = RankNormalizer(**kwargs)
        elif normalizer_type == "winsor":
            self.normalizers[name] = WinsorNormalizer(**kwargs)
        elif normalizer_type == "none":
            self.normalizers[name] = IdentityNormalizer()
        else:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}")

    def normalize_losses(self, losses: Dict[str, float]) -> Dict[str, float]:
        """Normalize all provided losses."""
        normalized = {}
        for name, value in losses.items():
            if name in self.normalizers:
                normalized[name] = self.normalizers[name](np.array([value]))[0]
            else:
                normalized[name] = value  # No normalization
        return normalized

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.state_dict() for k, v in self.normalizers.items()}

    def load_state_dict(self, state: Dict[str, Any]):
        for name, norm_state in state.items():
            if name in self.normalizers:
                self.normalizers[name].load_state_dict(norm_state)


class IdentityNormalizer(Normalizer):
    """No-op normalizer that returns values unchanged."""

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass


class AdaptiveNormalizer(Normalizer):
    """Adaptive normalizer that switches strategies based on data characteristics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.history: List[float] = []
        self.rank_norm = RankNormalizer(window_size)
        self.winsor_norm = WinsorNormalizer(window_size=window_size)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        # Add to history
        if isinstance(values, (int, float)):
            values = np.array([values])
        self.history.extend(values.flatten())

        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

        if len(self.history) < 10:
            return np.zeros_like(values)

        # Choose strategy based on data characteristics
        arr = np.array(self.history)

        # Check for heavy tails (high kurtosis)
        if len(arr) > 20:
            # Simple kurtosis estimate
            mean = np.mean(arr)
            std = np.std(arr)
            if std > 0:
                kurtosis = np.mean(((arr - mean) / std)**4)
                if kurtosis > 3.0:  # Heavy tails
                    return self.winsor_norm(values)

        # Default to rank normalization
        return self.rank_norm(values)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank_norm.state_dict(),
            "winsor": self.winsor_norm.state_dict(),
            "history": self.history
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.rank_norm.load_state_dict(state["rank"])
        self.winsor_norm.load_state_dict(state["winsor"])
        self.history = state["history"]
