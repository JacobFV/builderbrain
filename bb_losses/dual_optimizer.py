"""
Dual optimizer for multi-objective constraint satisfaction.

Manages Lagrangian multipliers and constraint targets for stable multi-objective training.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
import numpy as np
from bb_core.math_utils import ConstraintManager, RankNormalizer, WinsorNormalizer


class DualOptimizer(nn.Module):
    """
    Dual optimizer for constraint-based multi-objective training.

    Maintains Lagrangian multipliers (λ) for each constraint and updates them
    to keep normalized losses near their targets while minimizing the primary loss.
    """

    def __init__(
        self,
        constraint_configs: Dict[str, Dict[str, Any]],
        eta_lambda: float = 1e-2,
        lambda_max: float = 50.0,
        use_pcgrad: bool = True
    ):
        super().__init__()

        self.constraint_configs = constraint_configs
        # Ensure eta_lambda is a float (YAML may load as string)
        self.eta_lambda = float(eta_lambda) if not isinstance(eta_lambda, float) else eta_lambda
        # Ensure lambda_max is a float
        self.lambda_max = float(lambda_max) if not isinstance(lambda_max, float) else lambda_max
        self.use_pcgrad = use_pcgrad

        # Initialize constraint manager
        self.constraint_manager = ConstraintManager(eta_lambda, lambda_max)

        # Set up normalizers and targets for each constraint
        for name, config in constraint_configs.items():
            normalizer_type = config.get('normalizer', 'rank')
            normalizer_kwargs = config.get('normalizer_kwargs', {})

            if normalizer_type == 'rank':
                normalizer = RankNormalizer(**normalizer_kwargs)
            elif normalizer_type == 'winsor':
                normalizer = WinsorNormalizer(**normalizer_kwargs)
            else:
                normalizer = RankNormalizer()  # Default fallback

            self.constraint_manager.add_constraint(name, config['target'], normalizer)

        # Track gradient conflicts for PCGrad
        self.gradient_conflicts = []

    def compute_lagrangian(
        self,
        task_loss: torch.Tensor,
        constraint_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Lagrangian with current dual variables.

        Args:
            task_loss: Primary task loss
            constraint_losses: Dictionary of constraint loss values

        Returns:
            Total Lagrangian loss and normalized constraint values
        """
        # Normalize constraint losses
        normalized_losses = self.constraint_manager.normalize_losses(constraint_losses)

        # Compute Lagrangian contribution from constraints
        constraint_penalty = self.constraint_manager.get_lagrangian_contribution(constraint_losses)

        # Total loss
        total_loss = task_loss + constraint_penalty

        return total_loss, normalized_losses

    def update_duals(self, normalized_losses: Dict[str, torch.Tensor]):
        """Update dual variables based on normalized constraint losses."""
        # Convert to Python floats for the constraint manager
        dual_updates = {k: float(v) for k, v in normalized_losses.items()}
        self.constraint_manager.update_duals(dual_updates)

    def get_dual_values(self) -> Dict[str, float]:
        """Get current dual variable values."""
        return self.constraint_manager.duals.copy()

    def compute_gradient_conflicts(
        self,
        task_grad: torch.Tensor,
        constraint_grads: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute gradient conflict statistics for monitoring.

        Returns average cosine similarity between conflicting gradients.
        """
        conflicts = []

        for name, grad in constraint_grads.items():
            if grad is not None and task_grad is not None:
                # Compute cosine similarity
                dot_product = torch.dot(task_grad.flatten(), grad.flatten())
                norm_task = torch.norm(task_grad)
                norm_constraint = torch.norm(grad)

                if norm_task > 0 and norm_constraint > 0:
                    cosine = dot_product / (norm_task * norm_constraint)
                    conflicts.append(float(cosine))

        return np.mean(conflicts) if conflicts else 0.0

    def resolve_gradient_conflicts(
        self,
        task_loss: torch.Tensor,
        constraint_losses: Dict[str, torch.Tensor],
        task_grad_fn: Callable,
        constraint_grad_fns: Dict[str, Callable]
    ) -> torch.Tensor:
        """
        Resolve gradient conflicts using PCGrad or similar methods.

        This is a simplified version - in practice would implement full PCGrad.
        """
        if not self.use_pcgrad or len(constraint_losses) <= 1:
            # No conflict resolution needed
            total_loss = task_loss
            for loss_val in constraint_losses.values():
                total_loss = total_loss + loss_val
            return total_loss

        # Simplified conflict resolution: just sum losses
        # Full PCGrad implementation would project gradients
        total_loss = task_loss
        for loss_val in constraint_losses.values():
            total_loss = total_loss + loss_val

        return total_loss

    def log_training_metrics(self) -> Dict[str, Any]:
        """Log current training metrics for monitoring."""
        duals = self.get_dual_values()

        return {
            'dual_variables': duals,
            'constraint_targets': {
                name: config['target']
                for name, config in self.constraint_configs.items()
            },
            'gradient_conflicts': np.mean(self.gradient_conflicts) if self.gradient_conflicts else 0.0
        }

    def state_dict(self) -> Dict[str, Any]:
        """Save optimizer state."""
        return {
            'constraint_manager': self.constraint_manager.state_dict(),
            'gradient_conflicts': self.gradient_conflicts
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        self.constraint_manager.load_state_dict(state_dict['constraint_manager'])
        self.gradient_conflicts = state_dict.get('gradient_conflicts', [])


class AdaptiveDualOptimizer(DualOptimizer):
    """
    Adaptive dual optimizer that adjusts constraint targets based on training progress.

    Dynamically adjusts constraint targets to maintain optimal training dynamics.
    """

    def __init__(
        self,
        constraint_configs: Dict[str, Dict[str, Any]],
        eta_lambda: float = 1e-2,
        adaptation_rate: float = 0.01
    ):
        super().__init__(constraint_configs, eta_lambda)

        self.adaptation_rate = adaptation_rate
        self.violation_history: Dict[str, List[float]] = {
            name: [] for name in constraint_configs.keys()
        }

    def update_constraint_targets(self, normalized_losses: Dict[str, torch.Tensor]):
        """Adaptively update constraint targets based on violation patterns."""
        for name, loss_val in normalized_losses.items():
            if name not in self.constraint_configs:
                continue

            # Track violation history
            target = self.constraint_configs[name]['target']
            violation = float(loss_val - target)
            self.violation_history[name].append(violation)

            # Keep only recent history
            if len(self.violation_history[name]) > 1000:
                self.violation_history[name] = self.violation_history[name][-1000:]

            # Adaptive target adjustment
            if len(self.violation_history[name]) > 100:
                recent_violations = self.violation_history[name][-100:]

                # If consistently violating (too strict), relax target
                if np.mean(recent_violations) > 0.1:
                    new_target = min(target * 1.1, 1.0)
                    self.constraint_configs[name]['target'] = new_target

                # If consistently under-satisfying (too loose), tighten target
                elif np.mean(recent_violations) < -0.1:
                    new_target = max(target * 0.9, 0.01)
                    self.constraint_configs[name]['target'] = new_target

    def compute_loss_with_adaptation(
        self,
        task_loss: torch.Tensor,
        constraint_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss with adaptive target updates."""
        total_loss, normalized_losses = self.compute_lagrangian(task_loss, constraint_losses)

        # Update targets based on recent performance
        self.update_constraint_targets(normalized_losses)

        return total_loss, normalized_losses
