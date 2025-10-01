"""
Protocol definitions for builderbrain components.

Defines abstract interfaces that all implementations must follow.
"""

from typing import Dict, List, Any, Optional, Protocol, runtime_checkable, Tuple
from abc import ABC, abstractmethod
import numpy as np


@runtime_checkable
class GrammarParser(Protocol):
    """Protocol for grammar parsers that provide token masking."""

    def next_valid_tokens(self, prefix: str) -> List[int]:
        """Return list of valid token IDs that can follow the prefix."""
        ...

    def validate_sequence(self, tokens: List[int]) -> bool:
        """Check if token sequence is valid according to grammar."""
        ...

    def sequence_energy(self, tokens: List[int]) -> float:
        """Compute grammar compliance energy for sequence."""
        ...


@runtime_checkable
class PlanChecker(Protocol):
    """Protocol for plan validation and execution."""

    def validate_plan(self, plan_dag: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plan DAG against schema. Returns {'valid': bool, 'errors': [...]}"""
        ...

    def check_preconditions(self, plan_dag: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Check if plan preconditions hold in current state."""
        ...

    def estimate_resources(self, plan_dag: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource requirements for plan execution."""
        ...


@runtime_checkable
class ProgramAdapter(Protocol):
    """Protocol for program skill adapters."""

    def __call__(self, hidden_state: np.ndarray, skill_id: int) -> np.ndarray:
        """Apply skill adapter to hidden state."""
        ...

    def get_embedding(self, skill_id: int) -> np.ndarray:
        """Get embedding vector for skill."""
        ...

    def num_programs(self) -> int:
        """Return number of available programs."""
        ...


@runtime_checkable
class WorldModel(Protocol):
    """Protocol for world model used in planning."""

    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        """Encode observation into latent state."""
        ...

    def predict_next(self, state: np.ndarray, action: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Predict next state and associated metrics."""
        ...

    def rollout(self, start_state: np.ndarray, actions: List[Dict[str, Any]], horizon: int) -> List[Dict[str, Any]]:
        """Generate trajectory rollout."""
        ...

    def compute_evsi(self, state: np.ndarray, tool_action: Dict[str, Any], cost: float) -> float:
        """Compute expected value of sample information."""
        ...


@runtime_checkable
class SafetyMonitor(Protocol):
    """Protocol for safety monitoring and gating."""

    def compute_risk_energy(self, state: Dict[str, Any]) -> float:
        """Compute current risk energy V_s."""
        ...

    def check_promotion(self, candidate_state: Dict[str, Any], baseline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if candidate can be promoted. Returns {'approved': bool, 'reason': str}"""
        ...

    def log_safety_event(self, event: Dict[str, Any]) -> None:
        """Log safety-relevant event."""
        ...


class BaseModel(ABC):
    """Abstract base for neural network models."""

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass through model."""
        pass

    @abstractmethod
    def get_trainable_params(self) -> List[Any]:
        """Return list of trainable parameters."""
        pass

    @abstractmethod
    def load_weights(self, weights: Dict[str, Any]) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def save_weights(self) -> Dict[str, Any]:
        """Save model weights."""
        pass


class LossFunction(ABC):
    """Abstract base for loss functions."""

    @abstractmethod
    def __call__(self, predictions: Dict[str, Any], targets: Dict[str, Any]) -> float:
        """Compute loss value."""
        pass

    @abstractmethod
    def backward(self) -> Dict[str, np.ndarray]:
        """Compute gradients w.r.t. inputs."""
        pass


class ConstraintFunction(ABC):
    """Abstract base for constraint functions."""

    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> float:
        """Compute constraint violation."""
        pass

    @abstractmethod
    def get_target(self) -> float:
        """Return target value for constraint."""
        pass
