"""
Mathematical utilities for builderbrain core operations.

Includes normalization functions, gradient utilities, and constraint management.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from abc import ABC, abstractmethod


class Normalizer(ABC):
    """Base class for loss normalization strategies."""

    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Normalize input values."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return state for serialization."""
        pass

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from serialization."""
        pass


class RankNormalizer(Normalizer):
    """Rank-based normalization to [-1, 1] range."""

    def __init__(self, window_size: int = 50000):
        self.window_size = window_size
        self.history: List[float] = []

    def __call__(self, values: np.ndarray) -> np.ndarray:
        # Add new values to history
        if isinstance(values, (int, float)):
            values = np.array([values])
        self.history.extend(values.flatten())

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

        if len(self.history) < 2:
            return np.zeros_like(values)

        # Compute empirical CDF and map to [-1, 1]
        ranks = []
        for val in values.flatten():
            rank = np.sum(np.array(self.history) <= val) / len(self.history)
            normalized = 2 * rank - 1
            ranks.append(normalized)

        return np.array(ranks).reshape(values.shape)

    def state_dict(self) -> Dict[str, Any]:
        return {"window_size": self.window_size, "history": self.history}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.window_size = state["window_size"]
        self.history = state["history"]


class WinsorNormalizer(Normalizer):
    """Winsorized z-score normalization."""

    def __init__(self, tau: float = 3.0, window_size: int = 10000):
        self.tau = tau
        self.window_size = window_size
        self.history: List[float] = []
        self._mean = 0.0
        self._std = 1.0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        # Add new values to history
        if isinstance(values, (int, float)):
            values = np.array([values])
        self.history.extend(values.flatten())

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

        if len(self.history) < 2:
            return np.zeros_like(values)

        # Compute robust statistics
        arr = np.array(self.history)
        self._mean = np.median(arr)  # Use median for robustness
        self._std = np.std(arr)

        if self._std == 0:
            return np.zeros_like(values)

        # Winsorize and z-score
        clipped = np.clip(values, self._mean - self.tau * self._std,
                         self._mean + self.tau * self._std)
        return (clipped - self._mean) / self._std

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tau": self.tau,
            "window_size": self.window_size,
            "history": self.history,
            "mean": self._mean,
            "std": self._std
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.tau = state["tau"]
        self.window_size = state["window_size"]
        self.history = state["history"]
        self._mean = state["mean"]
        self._std = state["std"]


class GradientUtils:
    """Utilities for gradient conflict resolution."""

    @staticmethod
    def pcgrad(gradients: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        PCGrad: Resolve gradient conflicts via projection.

        Args:
            gradients: List of gradient vectors
            normalize: Whether to normalize gradients before projection

        Returns:
            Conflict-resolved gradient
        """
        if len(gradients) <= 1:
            return gradients[0] if gradients else np.array([])

        grads = [g.copy() for g in gradients]

        if normalize:
            grads = [g / (np.linalg.norm(g) + 1e-8) for g in grads]

        # Project each gradient onto the others
        for i in range(len(grads)):
            for j in range(len(grads)):
                if i != j:
                    dot = np.dot(grads[i], grads[j])
                    if dot < 0:  # Conflicting direction
                        proj = (dot / (np.linalg.norm(grads[j])**2 + 1e-8)) * grads[j]
                        grads[i] -= proj

        # Average the resolved gradients
        return np.mean(grads, axis=0)

    @staticmethod
    def gradient_cosine_matrix(gradients: List[np.ndarray]) -> np.ndarray:
        """Compute cosine similarity matrix between gradients."""
        if len(gradients) <= 1:
            return np.array([[1.0]])

        grads = [g / (np.linalg.norm(g) + 1e-8) for g in gradients]
        n = len(grads)
        cosines = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                cosines[i, j] = np.dot(grads[i], grads[j])

        return cosines


class ConstraintManager:
    """Manages dual variables for constraint satisfaction."""

    def __init__(self, eta_lambda: float = 1e-2, lambda_max: float = 50.0):
        self.eta_lambda = eta_lambda
        self.lambda_max = lambda_max
        self.duals: Dict[str, float] = {}
        self.normalizers: Dict[str, Normalizer] = {}

    def add_constraint(self, name: str, target: float, normalizer: Normalizer):
        """Add a new constraint with its target and normalizer."""
        self.duals[name] = 0.0
        self.normalizers[name] = normalizer

    def update_duals(self, raw_losses: Dict[str, float]):
        """Update dual variables based on current losses."""
        for name, raw_loss in raw_losses.items():
            if name not in self.duals:
                continue

            # Get target for this constraint
            target = getattr(self, '_targets', {}).get(name, 0.0)

            # Normalize the loss
            normalized = self.normalizers[name](np.array([raw_loss]))[0]

            # Update dual variable
            delta = normalized - target
            self.duals[name] = max(0.0, min(self.lambda_max,
                          self.duals[name] + self.eta_lambda * delta))

    def normalize_losses(self, raw_losses: Dict[str, float]) -> Dict[str, float]:
        """Normalize multiple constraint losses."""
        normalized = {}
        for name, raw_loss in raw_losses.items():
            if name in self.normalizers:
                normalized[name] = self.normalizers[name](np.array([raw_loss]))[0]
            else:
                normalized[name] = raw_loss  # No normalization if no normalizer set
        return normalized

    def get_lagrangian_contribution(self, raw_losses: Dict[str, float]) -> float:
        """Compute the Lagrangian constraint penalty."""
        total = 0.0
        normalized_losses = self.normalize_losses(raw_losses)

        for name, normalized_loss in normalized_losses.items():
            if name not in self.duals:
                continue

            target = getattr(self, '_targets', {}).get(name, 0.0)
            total += self.duals[name] * (normalized_loss - target)

        return total

    def state_dict(self) -> Dict[str, Any]:
        return {
            "duals": self.duals.copy(),
            "normalizers": {k: v.state_dict() for k, v in self.normalizers.items()}
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.duals = state["duals"]
        for name, norm_state in state["normalizers"].items():
            if name in self.normalizers:
                self.normalizers[name].load_state_dict(norm_state)
